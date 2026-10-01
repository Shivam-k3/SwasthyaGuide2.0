import '@testing-library/jest-dom/vitest'
import { vi } from 'vitest'

// jsdom does not implement these, and the application uses them for optional
// polish only. Stubbing them keeps tests focused on behaviour.
Object.defineProperty(window, 'scrollTo', {
  value: vi.fn(),
  writable: true,
})

// Not implemented by jsdom; the app calls it to bring a flagged field into view.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = vi.fn()
}

class MockIntersectionObserver implements IntersectionObserver {
  readonly root: Element | Document | null = null
  readonly rootMargin: string = ''
  readonly thresholds: ReadonlyArray<number> = []
  private readonly callback: IntersectionObserverCallback

  constructor(callback: IntersectionObserverCallback) {
    this.callback = callback
  }

  observe(target: Element): void {
    this.callback(
      [
        {
          isIntersecting: true,
          target,
          intersectionRatio: 1,
          boundingClientRect: target.getBoundingClientRect(),
          intersectionRect: target.getBoundingClientRect(),
          rootBounds: null,
          time: 0,
        } as IntersectionObserverEntry,
      ],
      this as unknown as IntersectionObserver,
    )
  }

  unobserve(): void {}
  disconnect(): void {}
  takeRecords(): IntersectionObserverEntry[] {
    return []
  }
}

vi.stubGlobal('IntersectionObserver', MockIntersectionObserver)
