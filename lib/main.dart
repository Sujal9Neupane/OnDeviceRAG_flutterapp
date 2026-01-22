import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_leap_sdk/flutter_leap_sdk.dart';
import 'package:http/http.dart' as http;
import 'package:onnxruntime/onnxruntime.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';
import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';

void main() {
  runApp(const LeapChatApp());
}

class LeapChatApp extends StatelessWidget {
  const LeapChatApp({super.key});

  @override
  Widget build(BuildContext context) {
    final colorScheme =
        ColorScheme.fromSeed(seedColor: const Color(0xFF1A6D64));
    return MaterialApp(
      title: 'Leap Chat',
      theme: ThemeData(colorScheme: colorScheme, useMaterial3: true),
      home: const ChatScreen(),
    );
  }
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatMessage {
  const _ChatMessage({required this.role, required this.text});

  final String role;
  final String text;
}

class _Course {
  const _Course({
    required this.id,
    required this.code,
    required this.name,
  });

  final int id;
  final String code;
  final String name;

  factory _Course.fromJson(Map<String, dynamic> json) {
    return _Course(
      id: json['id'] as int,
      code: (json['code'] as String?) ?? '',
      name: (json['name'] as String?) ?? '',
    );
  }
}

class _RetrievedChunk {
  const _RetrievedChunk({
    required this.id,
    required this.text,
    required this.score,
  });

  final int id;
  final String text;
  final double score;
}

class _ChatScreenState extends State<ChatScreen> {
  static const String _modelName = 'LFM2-700M';
  static const int _maxTokens = 128; //128 orginal
  static const String _backendBaseUrl = 'http://192.168.1.109:8000';
  static const String _basicAuthUser = '';
  static const String _basicAuthPassword = '';
  static const String _embeddingModelAssetPath =
      'assets/models/model_qint8_arm64.onnx';
  static const String _tokenizerAssetPath = 'assets/models/tokenizer.json';
  static const int _embeddingMaxLength = 512;
  final List<_ChatMessage> _messages = [];
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  Conversation? _conversation;
  bool _isModelReady = false;
  bool _isLoading = true;
  bool _isGenerating = false;
  String _status = 'Initializing LEAP SDK...';
  double? _downloadPercent;
  bool _isDownloadingKb = false;
  String? _kbStatus;
  String? _kbFilePath;
  bool _isLoadingCourses = false;
  List<_Course> _courses = [];
  int? _selectedCourseId;
  Database? _kbDatabase;
  bool _isEmbeddingReady = false;
  bool _isRetrieving = false;
  int _retrievalRequestId = 0;
  bool _multiTurnEnabled = false;
  OrtSession? _ortSession;
  WordPieceTokenizer? _tokenizer;
  List<_RetrievedChunk> _retrievedChunks = [];

  @override
  void initState() {
    super.initState();
    _initializeModel();
    _loadCourses();
    _initializeEmbedding();
  }

  @override
  void dispose() {
    FlutterLeapSdkService.cancelStreaming();
    _controller.dispose();
    _scrollController.dispose();
    _kbDatabase?.close();
    _ortSession?.release();
    OrtEnv.instance.release();
    super.dispose();
  }

  Future<void> _initializeModel() async {
    setState(() {
      _isLoading = true;
      _status = 'Checking local model...';
    });

    try {
      final models = await FlutterLeapSdkService.getDownloadedModels();
      final exists = models.any((model) => model.contains(_modelName));
      if (!exists) {
        setState(() {
          _status = 'Downloading $_modelName...';
          _downloadPercent = 0;
        });
        await FlutterLeapSdkService.downloadModel(
          modelName: _modelName,
          onProgress: (progress) {
            setState(() {
              _downloadPercent = progress.percentage;
            });
          },
        );
      }

      setState(() {
        _status = 'Loading $_modelName...';
      });
      await FlutterLeapSdkService.loadModel(modelPath: _modelName);
      if (_multiTurnEnabled) {
        _conversation = await FlutterLeapSdkService.createConversation(
          systemPrompt: _buildSystemPrompt(),
          generationOptions: const GenerationOptions(maxTokens: _maxTokens),
        );
      }

      setState(() {
        _isModelReady = true;
        _isLoading = false;
        _downloadPercent = null;
        _status = 'Model ready';
      });
    } catch (error) {
      setState(() {
        _isLoading = false;
        _status = 'Failed to initialize model: $error';
      });
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 200),
        curve: Curves.easeOut,
      );
    });
  }

  Future<void> _sendMessage() async {
    final text = _controller.text.trim();
    if (text.isEmpty || !_isModelReady || _isGenerating) return;

    setState(() {
      _controller.clear();
      _messages.add(_ChatMessage(role: 'user', text: text));
      _messages.add(const _ChatMessage(role: 'assistant', text: ''));
      _isGenerating = true;
    });
    _scrollToBottom();

    await _retrieveChunks(text);

    final assistantIndex = _messages.length - 1;

    try {
      final generationOptions = const GenerationOptions(maxTokens: _maxTokens);
      final systemPrompt = _buildSystemPrompt();
      final augmentedMessage = _buildAugmentedMessage(text);
      final stream = _multiTurnEnabled
          ? await _generateMultiTurnStream(
              augmentedMessage,
              generationOptions,
              systemPrompt,
            )
          : FlutterLeapSdkService.generateResponseStream(
              augmentedMessage,
              systemPrompt: systemPrompt,
              generationOptions: generationOptions,
            );
      stream.listen(
        (chunk) {
          setState(() {
            final existing = _messages[assistantIndex];
            _messages[assistantIndex] = _ChatMessage(
              role: existing.role,
              text: existing.text + chunk,
            );
          });
          _scrollToBottom();
        },
        onError: (error) {
          setState(() {
            _messages[assistantIndex] = _ChatMessage(
              role: 'assistant',
              text: 'Error: $error',
            );
            _isGenerating = false;
          });
        },
        onDone: () {
          setState(() {
            _isGenerating = false;
          });
        },
      );
    } catch (error) {
      setState(() {
        _messages[assistantIndex] = _ChatMessage(
          role: 'assistant',
          text: 'Error: $error',
        );
        _isGenerating = false;
      });
    }
  }

  Future<Stream<String>> _generateMultiTurnStream(
    String message,
    GenerationOptions generationOptions,
    String systemPrompt,
  ) async {
    if (_conversation == null) {
      _conversation = await FlutterLeapSdkService.createConversation(
        systemPrompt: systemPrompt,
        generationOptions: generationOptions,
      );
    } else {
      _conversation!.updateGenerationOptions(generationOptions);
    }
    return _conversation!.generateResponseStream(message);
  }

  Future<void> _toggleMultiTurn(bool value) async {
    setState(() {
      _multiTurnEnabled = value;
    });
    if (!_multiTurnEnabled) {
      _conversation = null;
      return;
    }
    if (_isModelReady) {
      _conversation = await FlutterLeapSdkService.createConversation(
        systemPrompt: _buildSystemPrompt(),
        generationOptions: const GenerationOptions(maxTokens: _maxTokens),
      );
    }
  }

  Future<void> _initializeEmbedding() async {
    try {
      final modelData = await rootBundle.load(_embeddingModelAssetPath);
      final tokenizerJson = await rootBundle.loadString(_tokenizerAssetPath);
      _tokenizer = _buildTokenizerFromJson(tokenizerJson);
      _ortSession = OrtSession.fromBuffer(
        modelData.buffer.asUint8List(),
        OrtSessionOptions(),
      );

      setState(() {
        _isEmbeddingReady = true;
      });
    } catch (error) {
      setState(() {
        _kbStatus = 'Embedding init failed: $error';
      });
    }
  }

  WordPieceTokenizer _buildTokenizerFromJson(String jsonString) {
    final data = jsonDecode(jsonString) as Map<String, dynamic>;
    final model = data['model'] as Map<String, dynamic>;
    final vocabList = model['vocab'] as List<dynamic>;
    final tokens = <String>[];
    for (final entry in vocabList) {
      if (entry is List && entry.isNotEmpty) {
        tokens.add(entry.first as String);
      }
    }

    void replaceToken(String from, String to) {
      final index = tokens.indexOf(from);
      if (index != -1) {
        tokens[index] = to;
      }
    }

    replaceToken('<s>', SpecialTokens.cls);
    replaceToken('</s>', SpecialTokens.sep);
    replaceToken('<pad>', SpecialTokens.pad);
    replaceToken('<unk>', SpecialTokens.unk);
    replaceToken('<mask>', SpecialTokens.mask);

    final vocab = Vocabulary.fromTokens(tokens);
    final tokenizer = WordPieceTokenizer(
      vocab: vocab,
      config: const WordPieceConfig(
        lowercase: false,
        stripAccents: false,
        handleChineseChars: false,
      ),
    )
      ..enableTruncation(maxLength: _embeddingMaxLength)
      ..enablePadding(length: _embeddingMaxLength);

    return tokenizer;
  }

  Future<void> _openKnowledgeBase(String dbPath) async {
    try {
      final database = await openDatabase(
        dbPath,
        readOnly: true,
      );
      setState(() {
        _kbDatabase = database;
      });
    } catch (error) {
      setState(() {
        _kbStatus = 'Failed to open KB: $error';
      });
    }
  }

  Future<void> _retrieveChunks(String query) async {
    if (!_isEmbeddingReady || _kbDatabase == null) {
      setState(() {
        _retrievedChunks = [];
      });
      return;
    }
    final requestId = ++_retrievalRequestId;

    setState(() {
      _isRetrieving = true;
      _retrievedChunks = [];
      _kbStatus = 'Retrieving relevant chunks...';
    });

    try {
      final embedding = await _embedQuery(query);
      final rows = await _kbDatabase!.query(
        'chunks',
        columns: ['id', 'text', 'vector'],
      );

      final top = <_RetrievedChunk>[];
      for (final row in rows) {
        final vectorRaw = row['vector'] as String?;
        if (vectorRaw == null || vectorRaw.isEmpty) continue;
        final vectorJson = jsonDecode(vectorRaw) as List<dynamic>;
        final vector = vectorJson.map((e) => (e as num).toDouble()).toList();
        if (vector.length != embedding.length) {
          continue;
        }
        final score = _cosineSimilarity(embedding, vector);
        final chunk = _RetrievedChunk(
          id: row['id'] as int,
          text: (row['text'] as String?) ?? '',
          score: score,
        );
        _insertTopK(top, chunk, 2);
      }

      setState(() {
        if (requestId != _retrievalRequestId) {
          return;
        }
        _retrievedChunks = top;
        _kbStatus = top.isEmpty
            ? 'No relevant chunks found.'
            : 'Retrieved top ${top.length} chunks.';
      });
    } catch (error) {
      setState(() {
        if (requestId != _retrievalRequestId) {
          return;
        }
        _kbStatus = 'Retrieval failed: $error';
      });
    } finally {
      setState(() {
        if (requestId != _retrievalRequestId) {
          return;
        }
        _isRetrieving = false;
      });
    }
  }

  String _buildSystemPrompt() {
    return 'You are a retrieval-augmented assistant. Use ONLY the provided context. '
        'If the answer is not in the context, say you do not know based on the provided context. '
        'Keep replies short.';
  }

  String _buildAugmentedMessage(String question) {
    final buffer = StringBuffer();
    buffer.writeln('Context:');
    if (_retrievedChunks.isEmpty) {
      buffer.writeln('No context available.');
    } else {
      for (var i = 0; i < _retrievedChunks.length; i++) {
        buffer.writeln('Source ${i + 1}: ${_retrievedChunks[i].text}');
      }
    }
    buffer.writeln('');
    buffer.writeln('Question: $question');
    return buffer.toString();
  }

  Future<List<double>> _embedQuery(String text) async {
    final tokenizer = _tokenizer!;
    final processed = _sentencePieceStyleText('query: $text');
    final encoding = tokenizer.encode(processed);
    final inputIds = Int64List.fromList(encoding.ids);
    final attentionMask = Int64List.fromList(encoding.attentionMask);
    final tokenTypeIds = Int64List.fromList(encoding.typeIds);

    final inputs = <String, OrtValueTensor>{
      'input_ids':
          OrtValueTensor.createTensorWithDataList(inputIds, [1, _embeddingMaxLength]),
      'attention_mask': OrtValueTensor.createTensorWithDataList(
        attentionMask,
        [1, _embeddingMaxLength],
      ),
    };
    if (_ortSession!.inputNames.contains('token_type_ids')) {
      inputs['token_type_ids'] = OrtValueTensor.createTensorWithDataList(
        tokenTypeIds,
        [1, _embeddingMaxLength],
      );
    }

    final outputs = _ortSession!.run(OrtRunOptions(), inputs);
    final output = outputs.first as OrtValueTensor;
    final values = output.value as List<dynamic>;
    final sequence = values.first as List<dynamic>;
    final firstToken = sequence.first as List<dynamic>;
    final hiddenSize = firstToken.length;
    final embedding = List<double>.filled(hiddenSize, 0);
    var tokenCount = 0;
    for (var i = 0; i < _embeddingMaxLength; i++) {
      if (attentionMask[i] == 0) continue;
      tokenCount += 1;
      final token = sequence[i] as List<dynamic>;
      for (var j = 0; j < hiddenSize; j++) {
        embedding[j] += (token[j] as num).toDouble();
      }
    }
    if (tokenCount > 0) {
      for (var j = 0; j < hiddenSize; j++) {
        embedding[j] /= tokenCount;
      }
    }
    return _l2Normalize(embedding);
  }

  String _sentencePieceStyleText(String text) {
    final parts = text.trim().split(RegExp(r'\s+'));
    final withMarkers = parts
        .where((part) => part.isNotEmpty)
        .map((part) => part.startsWith('▁') ? part : '▁$part')
        .join(' ');
    return withMarkers;
  }

  List<double> _l2Normalize(List<double> vector) {
    var sum = 0.0;
    for (final value in vector) {
      sum += value * value;
    }
    final norm = sqrt(sum);
    if (norm == 0) return vector;
    return vector.map((v) => v / norm).toList();
  }

  double _cosineSimilarity(List<double> a, List<double> b) {
    if (a.length != b.length) return -1;
    var dot = 0.0;
    var normA = 0.0;
    var normB = 0.0;
    for (var i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    final denom = sqrt(normA) * sqrt(normB);
    if (denom == 0) return -1;
    return dot / denom;
  }

  void _insertTopK(List<_RetrievedChunk> top, _RetrievedChunk chunk, int k) {
    top.add(chunk);
    top.sort((a, b) => b.score.compareTo(a.score));
    if (top.length > k) {
      top.removeRange(k, top.length);
    }
  }

  Future<void> _loadCourses() async {
    if (_isLoadingCourses) return;
    setState(() {
      _isLoadingCourses = true;
      _kbStatus = 'Loading courses...';
    });

    try {
      final uri =
          Uri.parse('$_backendBaseUrl/api/knowledge/courses/?page=1');
      final headers = <String, String>{};
      if (_basicAuthUser.isNotEmpty && _basicAuthPassword.isNotEmpty) {
        final token = base64Encode(
          utf8.encode('$_basicAuthUser:$_basicAuthPassword'),
        );
        headers['Authorization'] = 'Basic $token';
      }
      final response = await http.get(uri, headers: headers);
      if (response.statusCode != 200) {
        throw Exception('HTTP ${response.statusCode}: ${response.reasonPhrase}');
      }

      final payload = jsonDecode(response.body) as Map<String, dynamic>;
      final results = payload['results'] as List<dynamic>? ?? [];
      final courses = results
          .map((item) => _Course.fromJson(item as Map<String, dynamic>))
          .toList();

      setState(() {
        _courses = courses;
        _selectedCourseId = courses.isNotEmpty ? courses.first.id : null;
        _kbStatus = courses.isEmpty
            ? 'No courses found.'
            : 'Select a course to download the knowledge base.';
      });
      if (_selectedCourseId != null) {
        await _loadLocalKnowledgeBase(_selectedCourseId!);
      }
    } catch (error) {
      setState(() {
        _kbStatus = 'Failed to load courses: $error';
      });
    } finally {
      setState(() {
        _isLoadingCourses = false;
      });
    }
  }

  Future<void> _downloadKnowledgeBase() async {
    if (_isDownloadingKb) return;
    if (_selectedCourseId == null) {
      setState(() {
        _kbStatus = 'Select a course first.';
      });
      return;
    }

    setState(() {
      _isDownloadingKb = true;
      _kbStatus = 'Downloading knowledge base...';
    });

    try {
      final courseId = _selectedCourseId!;
      final uri = Uri.parse(
        '$_backendBaseUrl/api/knowledge/courses/$courseId/download_knowledge_base/',
      );
      final headers = <String, String>{};
      if (_basicAuthUser.isNotEmpty && _basicAuthPassword.isNotEmpty) {
        final token = base64Encode(
          utf8.encode('$_basicAuthUser:$_basicAuthPassword'),
        );
        headers['Authorization'] = 'Basic $token';
      }

      final response = await http.get(uri, headers: headers);
      if (response.statusCode != 200) {
        throw Exception('HTTP ${response.statusCode}: ${response.reasonPhrase}');
      }

      final directory = await getApplicationDocumentsDirectory();
      final file = File(_kbPathForCourse(directory.path, courseId));
      await file.writeAsBytes(response.bodyBytes, flush: true);

      setState(() {
        _kbFilePath = file.path;
        _kbStatus = 'Saved KB to ${file.path}';
      });
      await _openKnowledgeBase(file.path);
    } catch (error) {
      setState(() {
        _kbStatus = 'KB download failed: $error';
      });
    } finally {
      setState(() {
        _isDownloadingKb = false;
      });
    }
  }

  Future<void> _loadLocalKnowledgeBase(int courseId) async {
    final directory = await getApplicationDocumentsDirectory();
    final file = File(_kbPathForCourse(directory.path, courseId));
    final exists = await file.exists();
    if (!exists) {
      setState(() {
        _kbStatus = 'No local KB found for course $courseId.';
        _kbDatabase = null;
        _kbFilePath = null;
      });
      return;
    }
    setState(() {
      _kbFilePath = file.path;
      _kbStatus = 'Using local KB at ${file.path}';
    });
    await _openKnowledgeBase(file.path);
  }

  String _kbPathForCourse(String dirPath, int courseId) {
    return '$dirPath/knowledge_base_course_$courseId.db';
  }

  Widget _buildStatusBar(ThemeData theme) {
    if (!_isLoading) {
      return const SizedBox.shrink();
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      color: theme.colorScheme.surfaceContainerHighest,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(_status, style: theme.textTheme.bodyMedium),
          if (_downloadPercent != null) ...[
            const SizedBox(height: 8),
            LinearProgressIndicator(value: _downloadPercent! / 100),
          ] else ...[
            const SizedBox(height: 8),
            const LinearProgressIndicator(),
          ],
        ],
      ),
    );
  }

  Widget _buildCoursePicker(ThemeData theme) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
      child: Row(
        children: [
          Expanded(
            child: DropdownButton<int>(
              value: _selectedCourseId,
              isExpanded: true,
              hint: const Text('Select course'),
              items: _courses
                  .map(
                    (course) => DropdownMenuItem<int>(
                      value: course.id,
                      child: Text('${course.code} - ${course.name}'),
                    ),
                  )
                  .toList(),
              onChanged: _courses.isEmpty
                  ? null
                  : (value) {
                      setState(() {
                        _selectedCourseId = value;
                      });
                      if (value != null) {
                        _loadLocalKnowledgeBase(value);
                      }
                    },
            ),
          ),
          const SizedBox(width: 8),
          IconButton(
            onPressed: _isLoadingCourses ? null : _loadCourses,
            icon: _isLoadingCourses
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.refresh),
            tooltip: 'Refresh courses',
          ),
        ],
      ),
    );
  }

  Widget _buildRetrievedChunks(ThemeData theme) {
    if (_retrievedChunks.isEmpty) {
      return const SizedBox.shrink();
    }

    return Flexible(
      fit: FlexFit.loose,
      child: LayoutBuilder(
        builder: (context, constraints) {
          final maxHeight =
              min(200.0, constraints.maxHeight * 0.3).toDouble();
          return Container(
            width: double.infinity,
            margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: theme.colorScheme.tertiaryContainer,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Top retrieved chunks',
                  style: theme.textTheme.titleSmall?.copyWith(
                    color: theme.colorScheme.onTertiaryContainer,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 8),
                ConstrainedBox(
                  constraints: BoxConstraints(maxHeight: maxHeight),
                  child: SingleChildScrollView(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        for (var i = 0; i < _retrievedChunks.length; i++)
                          Padding(
                            padding: const EdgeInsets.only(bottom: 8),
                            child: Text(
                              'Source ${i + 1} (${_retrievedChunks[i].score.toStringAsFixed(3)}): ${_retrievedChunks[i].text}',
                              style: theme.textTheme.bodySmall?.copyWith(
                                color: theme.colorScheme.onTertiaryContainer,
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildMessageBubble(_ChatMessage message, ThemeData theme) {
    final isUser = message.role == 'user';
    final alignment = isUser ? Alignment.centerRight : Alignment.centerLeft;
    final bubbleColor = isUser
        ? theme.colorScheme.primaryContainer
        : theme.colorScheme.surfaceContainerHigh;
    final textColor = isUser
        ? theme.colorScheme.onPrimaryContainer
        : theme.colorScheme.onSurface;

    return Align(
      alignment: alignment,
      child: Container(
        margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        constraints: const BoxConstraints(maxWidth: 520),
        decoration: BoxDecoration(
          color: bubbleColor,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Text(message.text, style: TextStyle(color: textColor)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Leap Local Chat'),
        actions: [
          Row(
            children: [
              Text(
                'Multi-turn',
                style: theme.textTheme.labelMedium,
              ),
              Switch(
                value: _multiTurnEnabled,
                onChanged: (value) {
                  _toggleMultiTurn(value);
                },
              ),
            ],
          ),
          IconButton(
            onPressed: _isDownloadingKb || _selectedCourseId == null
                ? null
                : _downloadKnowledgeBase,
            icon: _isDownloadingKb
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.cloud_download),
            tooltip: 'Download knowledge base',
          ),
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: Center(
              child: Text(
                _isModelReady ? 'Ready' : 'Loading',
                style: theme.textTheme.labelMedium,
              ),
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          _buildStatusBar(theme),
          _buildCoursePicker(theme),
          _buildRetrievedChunks(theme),
          if (_kbStatus != null)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              color: theme.colorScheme.surfaceContainerLowest,
              child: Text(
                _kbStatus!,
                style: theme.textTheme.bodySmall,
              ),
            ),
          Expanded(
            child: _messages.isEmpty
                ? Center(
                    child: Text(
                      _isLoading
                          ? 'Preparing model...'
                          : 'Say hi to the local model.',
                      style: theme.textTheme.bodyLarge,
                    ),
                  )
                : ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    itemCount: _messages.length,
                    itemBuilder: (context, index) =>
                        _buildMessageBubble(_messages[index], theme),
                  ),
          ),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _controller,
                      textInputAction: TextInputAction.send,
                      onSubmitted: (_) => _sendMessage(),
                      decoration: const InputDecoration(
                        hintText: 'Message the model...',
                        border: OutlineInputBorder(),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  FilledButton(
                    onPressed:
                        _isModelReady && !_isGenerating ? _sendMessage : null,
                    child: _isGenerating
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Text('Send'),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
