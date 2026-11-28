import { afterEach, describe, expect, it, vi } from 'vitest'
import { fetchMetrics, predictFeatures } from './api'

const respond = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status })

afterEach(() => vi.unstubAllGlobals())

describe('API client', () => {
  it('falls back to the bundled training results when the metrics endpoint fails', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(respond({}, 404))
      .mockResolvedValueOnce(respond({ metrics: { rf_price_rmse: 1 } }))
    vi.stubGlobal('fetch', fetchMock)
    expect((await fetchMetrics()).metrics.rf_price_rmse).toBe(1)
    expect(fetchMock.mock.calls.map((call) => call[0])).toEqual(['/v1/metrics', '/train_results.json'])
  })

  it('posts scenario features and rejects HTTP errors', async () => {
    const fetchMock = vi.fn().mockResolvedValue(respond({}, 400))
    vi.stubGlobal('fetch', fetchMock)
    await expect(predictFeatures('US', { a: 1 })).rejects.toThrow('HTTP 400')
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ region: 'US', features: { a: 1 } })
  })
})
