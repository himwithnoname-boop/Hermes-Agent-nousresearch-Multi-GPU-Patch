import type { ReactElement } from 'react'

import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n/types'
import { Cpu, Monitor, Package, Zap } from '@/lib/icons'
import type { LocalHardware } from '@/types/hermes'

import { Pill, SettingsSection } from './primitives'

function memoryLabel(bytes: number): string {
  return `${(bytes / 1e9).toFixed(1)} GB (${(bytes / 2 ** 30).toFixed(1)} GiB)`
}

interface MemoryFactsProps {
  detected: number
  usable?: number
  used?: number | null
  copy: Translations['settings']['localModels']
}

function MemoryFacts({ detected, usable, used, copy }: MemoryFactsProps): ReactElement {
  return (
    <span className="inline-flex flex-wrap gap-x-3 gap-y-1 tabular-nums">
      <span>
        {copy.memoryDetected ?? 'Detected'}: {memoryLabel(detected)}
      </span>
      {usable != null && (
        <span title={copy.memoryUsableHint ?? 'Conservative model-planning budget after reserved headroom.'}>
          {copy.memoryUsable ?? 'Usable'}: {memoryLabel(usable)}
        </span>
      )}
      {used != null && (
        <span title={copy.memoryInUseHint ?? 'Includes model, runtime and desktop allocations.'}>
          {copy.memoryInUse ?? 'In use'}: {memoryLabel(used)}
        </span>
      )}
    </span>
  )
}

export interface LocalModelsHardwareSectionProps {
  hardware: LocalHardware | undefined
}

export function LocalModelsHardwareSection({ hardware }: LocalModelsHardwareSectionProps): ReactElement {
  const { t } = useI18n()
  const copy = t.settings.localModels

  return (
    <SettingsSection icon={Monitor} title={copy.hardwareTitle}>
      {hardware ? (
        <div className="grid gap-3 py-1 text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <Cpu className="size-3.5" />
            <span>{copy.vram('').trim()}</span>
            <MemoryFacts
              copy={copy}
              detected={hardware.vram_total_bytes}
              usable={hardware.vram_usable_bytes}
              used={hardware.vram_used_bytes}
            />
            {hardware.uma && <Pill>{copy.unifiedMemory}</Pill>}
          </div>
          {hardware.gpus?.map((gpu, index) => (
            <div className="grid gap-1" key={`${index}:${gpu.name}`}>
              <span className="inline-flex items-center gap-1.5">
                <Zap className="size-3.5" />
                GPU {index}: {gpu.name}
              </span>
              <MemoryFacts copy={copy} detected={gpu.total_bytes} usable={gpu.usable_bytes} used={gpu.used_bytes} />
            </div>
          ))}
          {!hardware.gpus?.length && hardware.gpu_name && <span>{hardware.gpu_name}</span>}
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <Package className="size-3.5" />
            <span>{copy.ram(memoryLabel(hardware.ram_total_bytes))}</span>
            <span>
              {copy.memoryInUse ?? 'In use'}: {memoryLabel(hardware.ram_total_bytes - hardware.ram_available_bytes)}
            </span>
          </div>
        </div>
      ) : (
        <p className="py-1 text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
          {copy.hardwareLoading}
        </p>
      )}
    </SettingsSection>
  )
}
