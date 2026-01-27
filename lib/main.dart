import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:fllama/fllama.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';
import 'package:syncfusion_flutter_pdf/pdf.dart';

void main() {
  runApp(const FllamaDemoApp());
}

class FllamaDemoApp extends StatelessWidget {
  const FllamaDemoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fllama Inference',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      home: const FllamaHomePage(),
    );
  }
}

class FllamaHomePage extends StatefulWidget {
  const FllamaHomePage({super.key});

  @override
  State<FllamaHomePage> createState() => _FllamaHomePageState();
}

class _FllamaHomePageState extends State<FllamaHomePage> {
  static const String _lfmModelUrl =
      'https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q4_K_M.gguf?download=true';
  static const String _hyperclovaxModelUrl =
      'https://huggingface.co/kexplo/HyperCLOVAX-SEED-Text-Instruct-1.5B-Q4_K_M-GGUF/resolve/main/hyperclovax-seed-text-instruct-1.5b-q4_k_m.gguf?download=true';
  static const String _bgeM3KoreanEmbedderUrl =
      'https://huggingface.co/hongkeon/bge-m3-korean-Q4_K_M-GGUF/resolve/main/bge-m3-korean-q4_k_m.gguf?download=true';
  static const String _e5TinyEmbedderUrl =
      'https://huggingface.co/exp-models/dragonkue-KoEn-E5-Tiny/resolve/main/ggml-model-q4_k_m.gguf?download=true';
  final TextEditingController _modelPathController = TextEditingController();
  final TextEditingController _modelUrlController = TextEditingController(
    text:
        _hyperclovaxModelUrl,
  );
  final TextEditingController _embedderModelPathController =
      TextEditingController();
  final TextEditingController _embedderModelUrlController =
      TextEditingController(
    text:
        _bgeM3KoreanEmbedderUrl,
  );
  final TextEditingController _embedderPoolingController =
      TextEditingController(text: '1');
  final TextEditingController _embedderNormalizeController =
      TextEditingController(text: '2');
  final TextEditingController _systemPromptController = TextEditingController(
    text:
        'You are a concise on-device RAG assistant. Answer clearly and briefly.',
  );
  final TextEditingController _promptController = TextEditingController(
    text:
        'Explain how on-device retrieval keeps data private while improving answers.',
  );
  final TextEditingController _maxTokensController = TextEditingController(
    text: '128',
  );
  final TextEditingController _contextController = TextEditingController(
    text: '512',
  );

  String _status = 'Idle';
  bool _isDownloading = false;
  double _downloadProgress = 0;
  bool _isGenerating = false;
  int? _requestId;
  Stopwatch? _stopwatch;
  Timer? _statsTimer;
  double _rssMb = 0;
  int _estimatedTokens = 0;
  double _tokensPerSecond = 0;
  final List<ChatMessage> _messages = [];
  String? _defaultModelPath;
  String? _defaultEmbedderModelPath;
  Database? _ragDb;
  bool _isIndexing = false;
  int _indexedChunks = 0;
  int _totalChunks = 0;
  int _ragChunkCount = 0;
  String _lastRetrievedContext = '';
  bool _retrievalDebugEnabled = false;
  String _retrievalDebugInfo = '';
  int _lastRetrievalRowCount = 0;
  int _lastRetrievedChunkCount = 0;
  int _lastRetrievedContextLength = 0;

  @override
  void initState() {
    super.initState();
    _prepareDefaultModelPaths();
    _initRagDb();
    _startStatsTimer();
  }

  @override
  void dispose() {
    if (_requestId != null) {
      fllamaCancelInference(_requestId!);
    }
    _modelPathController.dispose();
    _modelUrlController.dispose();
    _embedderModelPathController.dispose();
    _embedderModelUrlController.dispose();
    _embedderPoolingController.dispose();
    _embedderNormalizeController.dispose();
    _systemPromptController.dispose();
    _promptController.dispose();
    _maxTokensController.dispose();
    _contextController.dispose();
    _statsTimer?.cancel();
    super.dispose();
  }

  Future<void> _prepareDefaultModelPaths() async {
    final modelsDir = await _getModelsDirectory();

    _defaultModelPath =
        await _defaultPathForUrl(_modelUrlController.text.trim());
    if (_modelPathController.text.trim().isEmpty &&
        _defaultModelPath != null) {
      _modelPathController.text = _defaultModelPath!;
    }

    final embUrl = _embedderModelUrlController.text.trim();
    final embUri = Uri.tryParse(embUrl);
    if (embUri != null && embUri.pathSegments.isNotEmpty) {
      final embName =
          embUri.pathSegments.isNotEmpty ? embUri.pathSegments.last : 'embedder.gguf';
      final embPath = p.join(modelsDir.path, embName);
      _defaultEmbedderModelPath = embPath;
      if (_embedderModelPathController.text.trim().isEmpty) {
        _embedderModelPathController.text = embPath;
      }
    }

    setState(() {});
  }

  Future<String?> _defaultPathForUrl(String url) async {
    if (url.isEmpty) return null;
    final uri = Uri.tryParse(url);
    if (uri == null || uri.pathSegments.isEmpty) return null;
    final modelsDir = await _getModelsDirectory();
    final fileName =
        uri.pathSegments.isNotEmpty ? uri.pathSegments.last : 'model.gguf';
    return p.join(modelsDir.path, fileName);
  }

  Future<void> _applyModelUrl(String url) async {
    _modelUrlController.text = url;
    final defaultPath = await _defaultPathForUrl(url);
    setState(() {
      _defaultModelPath = defaultPath;
      if (defaultPath != null) {
        _modelPathController.text = defaultPath;
      }
    });
  }

  Future<void> _applyEmbedderUrl(String url) async {
    _embedderModelUrlController.text = url;
    final defaultPath = await _defaultPathForUrl(url);
    setState(() {
      _defaultEmbedderModelPath = defaultPath;
      if (defaultPath != null) {
        _embedderModelPathController.text = defaultPath;
      }
    });
  }

  Future<void> _initRagDb() async {
    _ragDb = await _openRagDatabase();
    await _refreshRagChunkCount();
  }

  Future<Directory> _getModelsDirectory() async {
    final documentsDir = await getApplicationDocumentsDirectory();
    final modelsDir = Directory(p.join(documentsDir.path, 'models'));
    if (!await modelsDir.exists()) {
      await modelsDir.create(recursive: true);
    }
    return modelsDir;
  }

  Future<void> _importModelToController(
      TextEditingController controller) async {
    _setStatus('Selecting model file...');
    final result = await FilePicker.platform.pickFiles(
      type: FileType.any,
      allowCompression: false,
    );
    if (result == null) {
      _setStatus('Import cancelled.');
      return;
    }

    final sourcePath = result.files.single.path;
    if (sourcePath == null) {
      _setStatus('Unable to access selected file.');
      return;
    }
    if (p.extension(sourcePath).toLowerCase() != '.gguf') {
      _setStatus('Please select a .gguf file.');
      return;
    }

    try {
      final modelsDir = await _getModelsDirectory();
      final fileName = p.basename(sourcePath);
      final destination = p.join(modelsDir.path, fileName);

      if (await File(destination).exists()) {
        setState(() {
          controller.text = destination;
          _status = 'Model already in app storage.';
        });
        return;
      }

      _setStatus('Copying model into app storage...');
      await File(sourcePath).copy(destination);

      setState(() {
        controller.text = destination;
        _status = 'Model copied to app storage.';
      });
    } catch (e) {
      _setStatus('Import failed: $e');
    }
  }

  Future<void> _importModel() async {
    await _importModelToController(_modelPathController);
  }

  Future<void> _importEmbedderModel() async {
    await _importModelToController(_embedderModelPathController);
  }

  Future<void> _testEmbedder() async {
    final embedderPath = await _ensureEmbedderAvailable();
    if (embedderPath == null) return;
    _setStatus('Testing embedder...');
    final testStart = Stopwatch()..start();
    final embedding = await _embedText('test embedding', isQuery: true);
    testStart.stop();
    if (embedding == null || embedding.isEmpty) {
      return;
    }
    _setStatus(
      'Embedder ok: ${embedding.length} dims in ${testStart.elapsedMilliseconds} ms',
    );
  }

  Future<String?> _downloadModel(
    Uri uri,
    String destination, {
    TextEditingController? pathController,
  }) async {
    if (_isDownloading) {
      _setStatus('Download already in progress.');
      return null;
    }

    setState(() {
      _isDownloading = true;
      _downloadProgress = 0;
      _status = 'Downloading model...';
    });

    try {
      final request = await HttpClient().getUrl(uri);
      final response = await request.close();
      if (response.statusCode != 200) {
        throw HttpException('HTTP ${response.statusCode}');
      }

      final contentLength = response.contentLength;
      final file = File(destination);
      final sink = file.openWrite();
      var received = 0;

      await for (final chunk in response) {
        received += chunk.length;
        sink.add(chunk);
        if (contentLength > 0) {
          setState(() {
            _downloadProgress = received / contentLength;
          });
        }
      }

      await sink.flush();
      await sink.close();

      setState(() {
        (pathController ?? _modelPathController).text = destination;
        _status = 'Download complete.';
      });
      return destination;
    } catch (e) {
      _setStatus('Download failed: $e');
      return null;
    } finally {
      setState(() {
        _isDownloading = false;
      });
    }
  }

  Future<void> _downloadModelFromUrl(
      TextEditingController urlController,
      TextEditingController pathController) async {
    final url = urlController.text.trim();
    if (url.isEmpty) {
      _setStatus('Provide a model URL to download.');
      return;
    }

    final uri = Uri.tryParse(url);
    if (uri == null || !uri.hasScheme) {
      _setStatus('Invalid model URL.');
      return;
    }

    final modelsDir = await _getModelsDirectory();
    final fileName =
        uri.pathSegments.isNotEmpty ? uri.pathSegments.last : 'model.gguf';
    final destination = pathController.text.trim().isNotEmpty
        ? pathController.text.trim()
        : p.join(modelsDir.path, fileName);

    await _downloadModel(uri, destination, pathController: pathController);
  }

  Future<void> _checkModelExists() async {
    final modelPath = _modelPathController.text.trim();
    if (modelPath.isEmpty) {
      _setStatus('Provide a GGUF model path to check.');
      return;
    }

    final exists = await File(modelPath).exists();
    _setStatus(
      exists ? 'Model found at: $modelPath' : 'Model not found at: $modelPath',
    );
  }

  Future<void> _checkEmbedderExists() async {
    final modelPath = _embedderModelPathController.text.trim();
    if (modelPath.isEmpty) {
      _setStatus('Provide an embedder model path to check.');
      return;
    }

    final exists = await File(modelPath).exists();
    _setStatus(
      exists
          ? 'Embedder found at: $modelPath'
          : 'Embedder not found at: $modelPath',
    );
  }

  Future<String?> _ensureModelAvailable() async {
    _setStatus('Checking model...');
    final configuredPath = _modelPathController.text.trim();
    if (configuredPath.isNotEmpty &&
        await File(configuredPath).exists()) {
      _setStatus('Model ready.');
      return configuredPath;
    }

    final url = _modelUrlController.text.trim();
    if (url.isEmpty) {
      _setStatus('Model URL missing. Add it in Settings.');
      return null;
    }

    final uri = Uri.tryParse(url);
    if (uri == null || !uri.hasScheme) {
      _setStatus('Invalid model URL.');
      return null;
    }

    final modelsDir = await _getModelsDirectory();
    final fileName =
        uri.pathSegments.isNotEmpty ? uri.pathSegments.last : 'model.gguf';
    final destination = configuredPath.isNotEmpty
        ? configuredPath
        : (_defaultModelPath ?? p.join(modelsDir.path, fileName));

    if (await File(destination).exists()) {
      setState(() {
        _modelPathController.text = destination;
      });
      _setStatus('Model ready.');
      return destination;
    }

    final downloadedPath = await _downloadModel(
      uri,
      destination,
      pathController: _modelPathController,
    );
    if (downloadedPath != null) {
      _setStatus('Model ready.');
    }
    return downloadedPath;
  }

  Future<String?> _ensureEmbedderAvailable() async {
    final configuredPath = _embedderModelPathController.text.trim();
    if (configuredPath.isNotEmpty && await File(configuredPath).exists()) {
      _setStatus('Embedder ready.');
      return configuredPath;
    }

    _setStatus('Checking embedder...');
    final url = _embedderModelUrlController.text.trim();
    if (url.isEmpty) {
      _setStatus('Embedder URL missing. Add it in Settings.');
      return null;
    }

    final uri = Uri.tryParse(url);
    if (uri == null || !uri.hasScheme) {
      _setStatus('Invalid embedder URL.');
      return null;
    }

    final modelsDir = await _getModelsDirectory();
    final fileName =
        uri.pathSegments.isNotEmpty ? uri.pathSegments.last : 'embedder.gguf';
    final destination = configuredPath.isNotEmpty
        ? configuredPath
        : (_defaultEmbedderModelPath ?? p.join(modelsDir.path, fileName));

    if (await File(destination).exists()) {
      setState(() {
        _embedderModelPathController.text = destination;
      });
      _setStatus('Embedder ready.');
      return destination;
    }

    final downloadedPath = await _downloadModel(
      uri,
      destination,
      pathController: _embedderModelPathController,
    );
    if (downloadedPath != null) {
      _setStatus('Embedder ready.');
    }
    return downloadedPath;
  }

  Future<void> _runInference() async {
    if (_isGenerating) return;

    var prompt = _promptController.text.trim();
    if (prompt.isEmpty) {
      _setStatus('Enter a prompt to generate.');
      return;
    }
    const maxPromptTokens = 1024;
    final truncatedPrompt = _truncateToMaxTokens(prompt, maxPromptTokens);
    if (truncatedPrompt.length != prompt.length) {
      prompt = truncatedPrompt;
      _setStatus('Prompt truncated to $maxPromptTokens tokens.');
    }

    if (_ragChunkCount == 0) {
      _setStatus('Upload and index a document before chatting.');
      return;
    }

    final maxTokens = int.tryParse(_maxTokensController.text.trim());
    final contextSize = int.tryParse(_contextController.text.trim());
    if (maxTokens == null || maxTokens <= 0) {
      _setStatus('Invalid max tokens value.');
      return;
    }
    if (contextSize == null || contextSize <= 0) {
      _setStatus('Invalid context size value.');
      return;
    }

    _setStatus('Preparing model...');
    final modelPath = await _ensureModelAvailable();
    if (modelPath == null) return;

    String retrievedContext = '';
    if (_ragChunkCount > 0) {
      retrievedContext = await _retrieveContext(
        prompt,
        maxChars: _contextCharBudget(contextSize),
      );
    }

    final maxContextTokens = max(128, contextSize - maxTokens);
    const minRagTokens = 64;
    final availableForSystem = max(0, maxContextTokens - minRagTokens);
    final systemPromptBudget = min(256, availableForSystem);
    final ragBudget = maxContextTokens - systemPromptBudget;
    var systemPrompt = _systemPromptController.text.trim();
    systemPrompt = _truncateToMaxTokens(systemPrompt, systemPromptBudget);
    if (retrievedContext.isNotEmpty) {
      retrievedContext = _truncateToMaxTokens(retrievedContext, ragBudget);
    }

    setState(() {
      _lastRetrievedContext = retrievedContext;
      _lastRetrievedContextLength = retrievedContext.length;
      _messages.add(ChatMessage(Role.user, prompt));
      _messages.add(ChatMessage(Role.assistant, ''));
      _isGenerating = true;
      _status = 'Running inference...';
      _estimatedTokens = 0;
      _tokensPerSecond = 0;
    });

    _stopwatch = Stopwatch()..start();

    var ragPrompt = retrievedContext.isEmpty
        ? systemPrompt
        : '$systemPrompt\n\nContext:\n$retrievedContext';
    ragPrompt = _truncateToMaxTokens(ragPrompt, maxContextTokens);

    final request = OpenAiRequest(
      modelPath: modelPath,
      maxTokens: maxTokens,
      contextSize: contextSize,
      numGpuLayers: 0,
      messages: [
        Message(Role.system, ragPrompt),
        Message(Role.user, prompt),
      ],
    );

    _requestId = await fllamaChat(request, (response, _json, done) {
      if (!_isGenerating) return;

      final elapsedMs = _stopwatch?.elapsedMilliseconds ?? 0;
      final estimatedTokens = _estimateTokens(response);
      final tokensPerSecond = elapsedMs > 0
          ? estimatedTokens / (elapsedMs / 1000)
          : 0.0;

      setState(() {
        if (_messages.isNotEmpty &&
            _messages.last.role == Role.assistant) {
          _messages.last.content = response;
        }
        _estimatedTokens = estimatedTokens;
        _tokensPerSecond = tokensPerSecond;
      });

      if (done) {
        _stopwatch?.stop();
        final elapsedMs = _stopwatch?.elapsedMilliseconds ?? 0;
        setState(() {
          _isGenerating = false;
          _requestId = null;
          _status = 'Done in ${elapsedMs} ms';
          _estimatedTokens = estimatedTokens;
          _tokensPerSecond = tokensPerSecond;
          if (_messages.isNotEmpty &&
              _messages.last.role == Role.assistant) {
            _messages.last.inferenceMs = elapsedMs;
          }
        });
      }
    });
  }

  void _cancelInference() {
    if (_requestId == null) return;
    fllamaCancelInference(_requestId!);
    setState(() {
      _isGenerating = false;
      _requestId = null;
      _status = 'Inference cancelled.';
      _tokensPerSecond = 0;
    });
  }

  void _setStatus(String message) {
    setState(() {
      _status = message;
    });
  }

  void _startStatsTimer() {
    _statsTimer?.cancel();
    _statsTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      final rss = ProcessInfo.currentRss;
      setState(() {
        _rssMb = rss / (1024 * 1024);
      });
    });
  }

  Future<Database> _openRagDatabase() async {
    final dbPath = await getDatabasesPath();
    final path = p.join(dbPath, 'rag_index.db');
    return openDatabase(
      path,
      version: 1,
      onCreate: (db, _version) async {
        await db.execute('''
          CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            norm REAL NOT NULL
          )
        ''');
      },
    );
  }

  Future<Database> _ensureRagDb() async {
    _ragDb ??= await _openRagDatabase();
    await _refreshRagChunkCount();
    return _ragDb!;
  }

  Future<void> _refreshRagChunkCount() async {
    if (_ragDb == null) return;
    final result =
        Sqflite.firstIntValue(await _ragDb!.rawQuery('SELECT COUNT(*) FROM chunks'));
    setState(() {
      _ragChunkCount = result ?? 0;
    });
  }

  List<String> _chunkText(String text,
      {int maxChars = 800, int overlap = 120}) {
    return _recursiveCharacterSplit(text,
        maxChars: maxChars, overlap: overlap);
  }

  String _formatEmbedInput(String text, {required bool isQuery}) {
    return isQuery ? 'query: $text' : 'passage: $text';
  }

  Float32List _toFloat32List(List<double> values) {
    final floatList = Float32List(values.length);
    for (var i = 0; i < values.length; i++) {
      floatList[i] = values[i].toDouble();
    }
    return floatList;
  }

  double _vectorNorm(Float32List values) {
    var sum = 0.0;
    for (final v in values) {
      sum += v * v;
    }
    return sqrt(sum);
  }

  double _dot(Float32List a, Float32List b) {
    var sum = 0.0;
    final len = min(a.length, b.length);
    for (var i = 0; i < len; i++) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  Float32List _blobToFloat32List(Uint8List bytes) {
    final aligned = Uint8List.fromList(bytes);
    return Float32List.view(aligned.buffer);
  }

  int _contextCharBudget(int contextSize) {
    final budget = contextSize * 2;
    return budget.clamp(400, 1200).toInt();
  }

  int _estimateTokens(String text) {
    if (text.isEmpty) return 0;
    return max(1, (text.length / 4).round());
  }

  String _truncateToMaxTokens(String text, int maxTokens) {
    if (text.isEmpty) return text;
    final maxChars = maxTokens * 4;
    if (text.length <= maxChars) return text;
    return text.substring(0, maxChars).trimRight();
  }

  String _formatStatsLine() {
    final mem = _rssMb.toStringAsFixed(1);
    final tps = _tokensPerSecond.toStringAsFixed(2);
    return 'RAM: ${mem}MB • Tokens/sec: $tps • Tokens: $_estimatedTokens';
  }

  String _truncateText(String text, int maxChars) {
    if (text.length <= maxChars) return text;
    final truncated = text.substring(0, maxChars);
    final lastSpace = truncated.lastIndexOf(RegExp(r'\s'));
    if (lastSpace > 0) {
      return truncated.substring(0, lastSpace).trim();
    }
    return truncated.trim();
  }

  Future<Float32List?> _embedText(String text, {required bool isQuery}) async {
    final embedderPath = await _ensureEmbedderAvailable();
    if (embedderPath == null) return null;

    final input = _formatEmbedInput(text, isQuery: isQuery);
    final poolingType =
        int.tryParse(_embedderPoolingController.text.trim()) ?? 1;
    final embdNormalize =
        int.tryParse(_embedderNormalizeController.text.trim()) ?? 2;
    _setStatus('Embedding...');
    try {
      final embedding = await fllamaEmbed(
        FllamaEmbeddingRequest(
          input: input,
          modelPath: embedderPath,
          contextSize: 512,
          numGpuLayers: 0,
          numThreads: 2,
          poolingType: poolingType,
          embdNormalize: embdNormalize,
        ),
      );
      if (embedding.isEmpty) {
        _setStatus('Embedding failed. Check embedder model path.');
        return null;
      }
      return _toFloat32List(embedding);
    } catch (e) {
      _setStatus('Embedding failed: $e');
      return null;
    }
  }

  Future<void> _importRagDocument() async {
    if (_isIndexing) return;

    final embedderPath = await _ensureEmbedderAvailable();
    if (embedderPath == null) return;

    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['txt', 'md', 'text', 'pdf'],
    );
    if (result == null) return;

    final sourcePath = result.files.single.path;
    if (sourcePath == null) return;

    final file = File(sourcePath);
    if (!await file.exists()) return;

    setState(() {
      _isIndexing = true;
      _indexedChunks = 0;
      _totalChunks = 0;
      _status = 'Extracting text...';
    });

    final receivePort = ReceivePort();
    _setStatus('Chunking document...');
    await Isolate.spawn(
      _extractChunksForIndexingIsolate,
      _ChunkingRequest(
        filePath: file.path,
        maxChars: 800,
        overlap: 120,
        sendPort: receivePort.sendPort,
      ),
    );

    List<String> chunks = [];
    await for (final message in receivePort) {
      if (message is _ChunkingProgress) {
        setState(() {
          _status = 'Chunking... ${message.count} chunks';
        });
      } else if (message is _ChunkingResult) {
        chunks = message.chunks;
        receivePort.close();
        break;
      }
    }
    if (chunks.isEmpty) {
      _setStatus('No content to index.');
      setState(() {
        _isIndexing = false;
      });
      return;
    }

    final db = await _ensureRagDb();
    setState(() {
      _indexedChunks = 0;
      _totalChunks = chunks.length;
      _status = 'Chunked ${chunks.length} chunks. Starting embeddings...';
    });

    for (final chunk in chunks) {
      final embedding = await _embedText(chunk, isQuery: false);
      if (embedding == null) {
        _setStatus('Indexing stopped (embedding failed).');
        break;
      }
      final norm = _vectorNorm(embedding);
      final bytes = embedding.buffer.asUint8List(0, embedding.lengthInBytes);
      await db.insert('chunks', {
        'content': chunk,
        'embedding': bytes,
        'norm': norm,
      });
      setState(() {
        _indexedChunks += 1;
        _status = 'Indexed $_indexedChunks / $_totalChunks chunks';
      });
    }

    await _refreshRagChunkCount();
    setState(() {
      _isIndexing = false;
    });
  }


  Future<void> _clearRagIndex() async {
    final db = await _ensureRagDb();
    await db.delete('chunks');
    await _refreshRagChunkCount();
    _setStatus('RAG index cleared.');
  }

  Future<String> _retrieveContext(String query, {required int maxChars}) async {
    if (_ragDb == null) return '';
    if (_ragChunkCount == 0) return '';

    final queryEmbedding = await _embedText(query, isQuery: true);
    if (queryEmbedding == null) {
      if (_retrievalDebugEnabled) {
        _retrievalDebugInfo = 'Retrieval debug: embed failed';
      }
      _lastRetrievalRowCount = 0;
      _lastRetrievedChunkCount = 0;
      return '';
    }

    final queryNorm = _vectorNorm(queryEmbedding);
    if (queryNorm == 0) {
      if (_retrievalDebugEnabled) {
        _retrievalDebugInfo = 'Retrieval debug: query norm=0';
      }
      _lastRetrievalRowCount = 0;
      _lastRetrievedChunkCount = 0;
      return '';
    }

    final rows = await _ragDb!.query('chunks');
    if (rows.isEmpty) {
      if (_retrievalDebugEnabled) {
        _retrievalDebugInfo = 'Retrieval debug: 0 rows';
      }
      _lastRetrievalRowCount = 0;
      _lastRetrievedChunkCount = 0;
      return '';
    }

    final scored = <_ScoredChunk>[];
    var skippedZeroNorm = 0;
    for (final row in rows) {
      final bytes = row['embedding'] as Uint8List;
      final embd = _blobToFloat32List(bytes);
      final norm = (row['norm'] as num).toDouble();
      if (norm == 0) {
        skippedZeroNorm += 1;
        continue;
      }
      final score = _dot(queryEmbedding, embd) / (queryNorm * norm);
      scored.add(_ScoredChunk(row['content'] as String, score));
    }

    if (_retrievalDebugEnabled) {
      final topScores = scored
          .map((c) => c.score)
          .toList()
        ..sort((a, b) => b.compareTo(a));
      final top3 = topScores.take(3).map((s) => s.toStringAsFixed(3)).join(', ');
      _retrievalDebugInfo =
          'Retrieval debug: rows=${rows.length}, scored=${scored.length}, '
          'zeroNorm=$skippedZeroNorm, top=[$top3], budget=$maxChars';
    }

    scored.sort((a, b) => b.score.compareTo(a.score));
    _lastRetrievalRowCount = rows.length;
    _lastRetrievedChunkCount = min(2, scored.length);
    final top = scored.take(2).map((c) => c.content).toList();
    final combined = top.join('\n\n');
    return _truncateText(combined, maxChars);
  }

  Widget _buildChatTab(ThemeData theme) {
    final needsIndex = _ragChunkCount == 0;
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text('Chat', style: theme.textTheme.titleLarge),
          const SizedBox(height: 8),
          Expanded(
            child: ListView(
              children: [
                if (needsIndex) ...[
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: theme.colorScheme.surfaceContainerHighest,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(
                        color: theme.colorScheme.outlineVariant,
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Upload a document to enable chat',
                          style: theme.textTheme.titleSmall,
                        ),
                        const SizedBox(height: 6),
                        Text(
                          'Go to Settings > RAG index and import a document. '
                          'Chat will unlock after indexing completes.',
                          style: theme.textTheme.bodySmall,
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 12),
                ],
                ...[
                  Text('Retrieved context', style: theme.textTheme.titleSmall),
                  const SizedBox(height: 6),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: theme.colorScheme.surfaceContainerHighest,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: SelectableText(
                      _ragChunkCount == 0
                          ? 'No documents indexed yet.'
                          : _lastRetrievedContext.isEmpty
                              ? 'No context retrieved for this query.'
                              : _lastRetrievedContext,
                    ),
                  ),
                  const SizedBox(height: 12),
                ],
                if (_messages.isEmpty)
                  Center(
                    child: Text(
                      'Start a conversation to see responses here.',
                      style: theme.textTheme.bodySmall,
                    ),
                  )
                else
                  ..._messages.map(
                    (message) => _buildChatBubble(message, theme),
                  ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _promptController,
            decoration: const InputDecoration(
              labelText: 'Message',
              border: OutlineInputBorder(),
            ),
            maxLines: 3,
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: [
              FilledButton(
                onPressed:
                    _isGenerating || _ragChunkCount == 0 ? null : _runInference,
                child: const Text('Generate'),
              ),
              TextButton(
                onPressed: _isGenerating ? _cancelInference : null,
                child: const Text('Stop'),
              ),
            ],
          ),
          const SizedBox(height: 8),
          _buildStatusPanel(theme),
        ],
      ),
    );
  }

  Widget _buildPromptsTab(ThemeData theme) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text('Prompts', style: theme.textTheme.titleLarge),
        const SizedBox(height: 8),
        TextField(
          controller: _systemPromptController,
          decoration: const InputDecoration(
            labelText: 'System prompt',
            border: OutlineInputBorder(),
          ),
          maxLines: 5,
        ),
        const SizedBox(height: 16),
        _buildStatusPanel(theme),
      ],
    );
  }

  Widget _buildSettingsTab(ThemeData theme) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text('Model setup', style: theme.textTheme.titleLarge),
        const SizedBox(height: 8),
        TextField(
          controller: _modelPathController,
          decoration: const InputDecoration(
            labelText: 'GGUF model path',
            helperText: 'Defaults to the app models folder.',
            border: OutlineInputBorder(),
          ),
        ),
        if (_defaultModelPath != null) ...[
          const SizedBox(height: 6),
          Text(
            'Default: $_defaultModelPath',
            style: theme.textTheme.bodySmall,
          ),
        ],
        const SizedBox(height: 12),
        TextField(
          controller: _modelUrlController,
          decoration: const InputDecoration(
            labelText: 'Model download URL',
            helperText: 'Used automatically if the model is missing.',
            border: OutlineInputBorder(),
          ),
        ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 12,
          runSpacing: 12,
          children: [
            OutlinedButton(
              onPressed: () => _applyModelUrl(_lfmModelUrl),
              child: const Text('Use LFM2.5-1.2B'),
            ),
            OutlinedButton(
              onPressed: () => _applyModelUrl(_hyperclovaxModelUrl),
              child: const Text('Use HyperCLOVAX 1.5B'),
            ),
            OutlinedButton(
              onPressed: _importModel,
              child: const Text('Import GGUF'),
            ),
            OutlinedButton(
              onPressed: _checkModelExists,
              child: const Text('Check model'),
            ),
            FilledButton(
              onPressed: _isDownloading
                  ? null
                  : () => _downloadModelFromUrl(
                        _modelUrlController,
                        _modelPathController,
                      ),
              child: const Text('Download model'),
            ),
            Text(
              'CPU-only (numGpuLayers = 0)',
              style: theme.textTheme.bodySmall,
            ),
          ],
        ),
        if (_isDownloading) ...[
          const SizedBox(height: 12),
          LinearProgressIndicator(value: _downloadProgress),
        ],
        const SizedBox(height: 24),
        Text('Embedding model', style: theme.textTheme.titleLarge),
        const SizedBox(height: 8),
        TextField(
          controller: _embedderModelPathController,
          decoration: const InputDecoration(
            labelText: 'Embedder GGUF path',
            helperText: 'Used for retrieval embeddings.',
            border: OutlineInputBorder(),
          ),
        ),
        if (_defaultEmbedderModelPath != null) ...[
          const SizedBox(height: 6),
          Text(
            'Default: $_defaultEmbedderModelPath',
            style: theme.textTheme.bodySmall,
          ),
        ],
        const SizedBox(height: 12),
        TextField(
          controller: _embedderModelUrlController,
          decoration: const InputDecoration(
            labelText: 'Embedder download URL',
            border: OutlineInputBorder(),
          ),
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: TextField(
                controller: _embedderPoolingController,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: 'Pooling type',
                  border: OutlineInputBorder(),
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: TextField(
                controller: _embedderNormalizeController,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: 'Normalize',
                  border: OutlineInputBorder(),
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 12,
          runSpacing: 12,
          children: [
            OutlinedButton(
              onPressed: () => _applyEmbedderUrl(_bgeM3KoreanEmbedderUrl),
              child: const Text('Use BGE-M3 Korean Q4_K_M'),
            ),
            OutlinedButton(
              onPressed: () => _applyEmbedderUrl(_e5TinyEmbedderUrl),
              child: const Text('Use E5 Tiny'),
            ),
            OutlinedButton(
              onPressed: _importEmbedderModel,
              child: const Text('Import embedder'),
            ),
            OutlinedButton(
              onPressed: _checkEmbedderExists,
              child: const Text('Check embedder'),
            ),
            OutlinedButton(
              onPressed: _testEmbedder,
              child: const Text('Test embedder'),
            ),
            FilledButton(
              onPressed: _isDownloading
                  ? null
                  : () => _downloadModelFromUrl(
                        _embedderModelUrlController,
                        _embedderModelPathController,
                      ),
              child: const Text('Download embedder'),
            ),
          ],
        ),
        const SizedBox(height: 24),
        Text('RAG index', style: theme.textTheme.titleLarge),
        const SizedBox(height: 8),
        Text(
          'Indexed chunks: $_ragChunkCount',
          style: theme.textTheme.bodySmall,
        ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 12,
          runSpacing: 12,
          children: [
            FilledButton(
              onPressed: _isIndexing ? null : _importRagDocument,
              child: const Text('Import document'),
            ),
            TextButton(
              onPressed: _isIndexing ? null : _clearRagIndex,
              child: const Text('Clear index'),
            ),
            if (_isIndexing)
              Text(
                'Indexing $_indexedChunks / $_totalChunks...',
                style: theme.textTheme.bodySmall,
              ),
          ],
        ),
        const SizedBox(height: 16),
        SwitchListTile(
          contentPadding: EdgeInsets.zero,
          title: const Text('Retrieval debug'),
          subtitle: const Text('Show retrieval scoring details in status.'),
          value: _retrievalDebugEnabled,
          onChanged: (value) {
            setState(() {
              _retrievalDebugEnabled = value;
              _retrievalDebugInfo = '';
            });
          },
        ),
        const SizedBox(height: 24),
        Text('Generation settings', style: theme.textTheme.titleLarge),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: TextField(
                controller: _maxTokensController,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: 'Max tokens',
                  border: OutlineInputBorder(),
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: TextField(
                controller: _contextController,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: 'Context size',
                  border: OutlineInputBorder(),
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 16),
        _buildStatusPanel(theme),
      ],
    );
  }

  Widget _buildStatusPanel(ThemeData theme) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Status: $_status', style: theme.textTheme.bodySmall),
        const SizedBox(height: 6),
        Text(_formatStatsLine(), style: theme.textTheme.bodySmall),
        const SizedBox(height: 6),
        if (_isIndexing || _totalChunks > 0)
          Text(
            'Indexing: $_indexedChunks / $_totalChunks',
            style: theme.textTheme.bodySmall,
          ),
        if (_ragChunkCount > 0) ...[
          const SizedBox(height: 6),
          Text(
            'Retrieved: $_lastRetrievedChunkCount / $_lastRetrievalRowCount '
            'chunks • ${_lastRetrievedContextLength} chars',
            style: theme.textTheme.bodySmall,
          ),
        ],
        if (_retrievalDebugEnabled && _retrievalDebugInfo.isNotEmpty) ...[
          const SizedBox(height: 6),
          Text(_retrievalDebugInfo, style: theme.textTheme.bodySmall),
        ],
      ],
    );
  }

  Widget _buildChatBubble(ChatMessage message, ThemeData theme) {
    final isUser = message.role == Role.user;
    final bubbleColor = isUser
        ? theme.colorScheme.primaryContainer
        : theme.colorScheme.surfaceContainerHighest;
    final textColor = isUser
        ? theme.colorScheme.onPrimaryContainer
        : theme.colorScheme.onSurfaceVariant;

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 560),
        child: Container(
          margin: const EdgeInsets.symmetric(vertical: 6),
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: bubbleColor,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Column(
            crossAxisAlignment:
                isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
            children: [
              SelectableText(
                message.content.isEmpty ? '...' : message.content,
                style: theme.textTheme.bodyMedium?.copyWith(color: textColor),
              ),
              if (!isUser && message.inferenceMs != null) ...[
                const SizedBox(height: 6),
                Text(
                  'Inference: ${message.inferenceMs} ms',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: textColor.withOpacity(0.7),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return DefaultTabController(
      length: 3,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Fllama Inference'),
          actions: [
            Padding(
              padding: const EdgeInsets.only(right: 12),
              child: Align(
                alignment: Alignment.centerRight,
                child: TabBar(
                  isScrollable: true,
                  tabs: const [
                    Tab(text: 'Chat'),
                    Tab(text: 'Prompts'),
                    Tab(text: 'Settings'),
                  ],
                ),
              ),
            ),
          ],
        ),
        body: TabBarView(
          children: [
            _buildChatTab(theme),
            _buildPromptsTab(theme),
            _buildSettingsTab(theme),
          ],
        ),
      ),
    );
  }
}

class ChatMessage {
  ChatMessage(this.role, this.content, {this.inferenceMs});

  final Role role;
  String content;
  int? inferenceMs;
}

class _ScoredChunk {
  _ScoredChunk(this.content, this.score);

  final String content;
  final double score;
}

class _ChunkingRequest {
  const _ChunkingRequest({
    required this.filePath,
    required this.maxChars,
    required this.overlap,
    required this.sendPort,
  });

  final String filePath;
  final int maxChars;
  final int overlap;
  final SendPort sendPort;
}

class _ChunkingProgress {
  const _ChunkingProgress(this.count);

  final int count;
}

class _ChunkingResult {
  const _ChunkingResult(this.chunks);

  final List<String> chunks;
}

void _extractChunksForIndexingIsolate(_ChunkingRequest request) {
  final chunks = _extractChunksForIndexing(
    request.filePath,
    maxChars: request.maxChars,
    overlap: request.overlap,
    sendPort: request.sendPort,
  );
  request.sendPort.send(_ChunkingResult(chunks));
}

List<String> _extractChunksForIndexing(
  String path, {
  int maxChars = 800,
  int overlap = 120,
  SendPort? sendPort,
}) {
  final extension = p.extension(path).toLowerCase();
  String text;
  if (extension == '.pdf') {
    final bytes = File(path).readAsBytesSync();
    final document = PdfDocument(inputBytes: bytes);
    final extractor = PdfTextExtractor(document);
    text = extractor.extractText();
    document.dispose();
  } else {
    text = File(path).readAsStringSync();
  }

  if (text.trim().isEmpty) {
    return <String>[];
  }

  final chunks = _recursiveCharacterSplit(text,
      maxChars: maxChars, overlap: overlap);
  if (sendPort != null) {
    for (var i = 0; i < chunks.length; i++) {
      if ((i + 1) % 10 == 0) {
        sendPort.send(_ChunkingProgress(i + 1));
      }
    }
  }
  return chunks;
}

List<String> _recursiveCharacterSplit(
  String text, {
  int maxChars = 800,
  int overlap = 120,
}) {
  final separators = ['\n\n', '\n', '. ', ' ', ''];
  final splits =
      _recursiveSplitWithSeparators(text, separators, maxChars);
  return _mergeSplits(splits, maxChars, overlap);
}

List<String> _recursiveSplitWithSeparators(
  String text,
  List<String> separators,
  int maxChars,
) {
  if (text.length <= maxChars) {
    return [text];
  }
  if (separators.isEmpty) {
    return _hardSplit(text, maxChars);
  }
  final sep = separators.first;
  final rest = separators.sublist(1);
  if (sep.isEmpty) {
    return _hardSplit(text, maxChars);
  }
  final parts = _splitBySeparator(text, sep);
  final results = <String>[];
  for (final part in parts) {
    if (part.length <= maxChars) {
      results.add(part);
    } else {
      results.addAll(
          _recursiveSplitWithSeparators(part, rest, maxChars));
    }
  }
  return results;
}

List<String> _splitBySeparator(String text, String sep) {
  final parts = text.split(sep);
  final results = <String>[];
  for (var i = 0; i < parts.length; i++) {
    var part = parts[i];
    if (part.isEmpty) continue;
    if (i < parts.length - 1) {
      part = '$part$sep';
    }
    results.add(part);
  }
  return results;
}

List<String> _mergeSplits(
  List<String> splits,
  int maxChars,
  int overlap,
) {
  final chunks = <String>[];
  var current = '';
  for (final split in splits) {
    if (split.isEmpty) continue;
    if (current.isEmpty) {
      current = split;
      continue;
    }
    if (current.length + split.length <= maxChars) {
      current += split;
      continue;
    }
    final trimmed = current.trim();
    if (trimmed.isNotEmpty) {
      chunks.add(trimmed);
    }
    final overlapText = overlap > 0 && current.length > overlap
        ? current.substring(current.length - overlap)
        : (overlap > 0 ? current : '');
    current = overlapText + split;
  }
  final trimmed = current.trim();
  if (trimmed.isNotEmpty) {
    chunks.add(trimmed);
  }
  return chunks;
}

List<String> _hardSplit(String text, int maxChars) {
  final chunks = <String>[];
  var start = 0;
  while (start < text.length) {
    final end = min(start + maxChars, text.length);
    chunks.add(text.substring(start, end));
    if (end == text.length) break;
    start = end;
  }
  return chunks;
}
