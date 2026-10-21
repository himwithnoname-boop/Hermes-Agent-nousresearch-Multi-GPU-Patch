import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { LocalHardware } from '@/types/hermes'

import { LocalModelsHardwareSection } from './local-models-hardware-section'

const hardware: LocalHardware = {
  uma: false,
  vram_total_bytes: 92136 * 2 ** 20,
  vram_usable_bytes: 80 * 2 ** 30,
  ram_total_bytes: 64 * 2 ** 30,
  ram_available_bytes: 32 * 2 ** 30,
  vram_label: '90.0 GB',
  gpu_name: 'Two workstation GPUs',
  gpu_util_percent: null,
  vram_used_bytes: 24 * 2 ** 30,
  gpus: [0, 1].map(() => ({
    name: 'NVIDIA Example CUDA GPU',
    total_bytes: 46068 * 2 ** 20,
    usable_bytes: 40 * 2 ** 30,
    free_bytes: 30 * 2 ** 30,
    used_bytes: 12 * 2 ** 30
  }))
}

afterEach(cleanup)

describe('Local Models hardware memory units', () => {
  it('labels dedicated GPU and system memory as binary units from physical byte totals', async () => {
    await act(async () => {
      render(
        <I18nProvider>
          <LocalModelsHardwareSection hardware={hardware} />
        </I18nProvider>
      )
    })

    expect(screen.getByText('Detected: 96.6 GB (90.0 GiB)')).toBeTruthy()
    expect(screen.getByText('Usable: 85.9 GB (80.0 GiB)')).toBeTruthy()
    expect(screen.getByText('In use: 25.8 GB (24.0 GiB)')).toBeTruthy()
    expect(screen.getAllByText('Detected: 48.3 GB (45.0 GiB)')).toHaveLength(2)
    expect(screen.getAllByText('Usable: 42.9 GB (40.0 GiB)')).toHaveLength(2)
    expect(screen.getAllByText('In use: 12.9 GB (12.0 GiB)')).toHaveLength(2)
    expect(screen.getByText(/GPU 0: NVIDIA Example CUDA GPU/)).toBeTruthy()
    expect(screen.getByText(/GPU 1: NVIDIA Example CUDA GPU/)).toBeTruthy()
    expect(screen.getByText(/64\.0 GiB/)).toBeTruthy()
    expect(screen.queryByText(/90\.0 GB/)).toBeNull()
  })
})
