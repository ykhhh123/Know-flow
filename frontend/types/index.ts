export interface CitationSource {
  citationNumber: number;
  chunkId: string;
  documentId: string;
  documentTitle: string;
  chunkIndex: number;
  pageNumber?: number | null;
  sectionTitle?: string | null;
  snippet: string;
  retrievalScore?: number | null;
  sourceUrl: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: string;
  reasoningContent?: string;
  citations?: CitationSource[];
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
}

export interface Settings {
  openaiApiKey: string;
  model: string;
}
