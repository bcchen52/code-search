// Answer stream event payloads. These mirror cqa/trace.py and cqa/generate/answer.py;
// change them together.

export type Source = {
  label: string; // "C1"
  path: string;
  start_line: number;
  end_line: number;
  rerank_score: number | null;
};

export type CitationStatus = "valid" | "fabricated" | "out_of_range";

export type SourcesEvent = { type: "sources"; data: { sources: Source[]; low_confidence: boolean } };
export type TokenEvent = { type: "token"; data: { text: string } };
export type CitationsEvent = {
  type: "citations";
  data: {
    citations: { label: string; lines: [number, number] | null; sentence: number; status: CitationStatus }[];
    uncited_sentences: number[];
    malformed: string[]; // citation attempts that broke the grammar, e.g. "[C3:95-103]"
  };
};
export type DoneEvent = {
  type: "done";
  data: {
    trace: unknown;
    timings_ms: Record<string, number>;
    cost_usd: number;
    config_hash: string;
    cached: boolean;
    model: string; // the model that answered; differs from the configured one after a fallback
    stop_reason: string | null; // "refusal": discard the streamed text; "max_tokens": the answer was cut off
    query_id: number; // the stored query, for GET /queries/{id} and the retrieval inspector
  };
};
export type ErrorEvent = { type: "error"; data: { stage: string; message: string } };

export type AnswerEvent = SourcesEvent | TokenEvent | CitationsEvent | DoneEvent | ErrorEvent;
