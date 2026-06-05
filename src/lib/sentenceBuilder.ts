import type { Token } from './types';

/**
 * Consumes the token stream from the motion-gate router and maintains a live
 * transcript:
 *  - consecutive `letter` tokens concatenate into one spelled word
 *  - a `word` token is inserted as a whole word (flushing any pending letters)
 *  - a `boundary` token finalizes the current word (space)
 */
export class SentenceBuilder {
  private words: string[] = [];
  private pendingLetters = '';
  private history: Token[] = [];

  private flushLetters() {
    if (this.pendingLetters.length > 0) {
      this.words.push(this.pendingLetters);
      this.pendingLetters = '';
    }
  }

  push(token: Token) {
    this.history.push(token);
    switch (token.kind) {
      case 'letter':
        this.pendingLetters += token.value;
        break;
      case 'word':
        this.flushLetters();
        this.words.push(token.value);
        break;
      case 'boundary':
        this.flushLetters();
        break;
    }
  }

  /** Current raw transcript (ASL gloss style: tokens joined by spaces). */
  get transcript(): string {
    const parts = [...this.words];
    if (this.pendingLetters) parts.push(this.pendingLetters);
    return parts.join(' ').trim();
  }

  get tokens(): Token[] {
    return this.history;
  }

  /** Remove the last character of the pending word, or the last whole word. */
  backspace() {
    if (this.pendingLetters.length > 0) {
      this.pendingLetters = this.pendingLetters.slice(0, -1);
    } else if (this.words.length > 0) {
      this.words.pop();
    }
    this.history.pop();
  }

  clear() {
    this.words = [];
    this.pendingLetters = '';
    this.history = [];
  }
}
