<script setup>
import HeroSection from './components/HeroSection.vue'
import PageSection from './components/PageSection.vue'
import StatCards from './components/StatCards.vue'
import FailureModePie from './components/FailureModePie.vue'
import FigurePanel from './components/FigurePanel.vue'
import PerformanceTable from './components/PerformanceTable.vue'
import InteractiveFrame from './components/InteractiveFrame.vue'
import CitationBlock from './components/CitationBlock.vue'
import SiteFooter from './components/SiteFooter.vue'

const assetUrl = (path) => `${import.meta.env.BASE_URL}${path}`

const stats = [
  { value: '5,383', label: 'Paired Clinical Cases' },
  { value: '49', label: 'Diseases' },
  { value: '17', label: 'Evaluated LLMs' },
  { value: 'Bias Trap Rate', label: 'Core Metric' },
]

const ecrCards = [
  {
    title: 'Dynamic Causal Inference (DCI)',
    body: 'Structured diagnostic reasoning with dual-pathway perception, causal graph reasoning, and evidence audit.',
  },
  {
    title: 'Critic-Driven Graph & Memory Evolution (CGME)',
    body: 'Accumulates validated reasoning paths and evolves illness graphs.',
  },
]
</script>

<template>
  <div class="site-shell">
    <HeroSection />

    <PageSection id="abstract" title="Abstract">
      <p>
        Despite high accuracy on medical benchmarks, LLMs can exhibit the
        Einstellung Effect in clinical diagnosis by relying on statistical
        shortcuts rather than patient-specific evidence, causing misdiagnosis in
        atypical cases. Existing benchmarks are not designed to detect this
        failure mode, especially in differential diagnosis where subtle
        discriminative findings can determine the correct disease.
      </p>
      <p>
        MedEinst is a counterfactual benchmark with 5,383 paired clinical cases
        across 49 diseases. Each pair contains a control case and a trap case
        whose altered discriminative evidence flips the diagnosis. Susceptibility
        is measured with Bias Trap Rate, the probability of misdiagnosing trap
        cases despite correctly diagnosing controls. The paper evaluates 17 LLMs
        and proposes ECR-Agent, aligning reasoning with evidence-based medicine
        through Dynamic Causal Inference (DCI) and Critic-Driven Graph & Memory
        Evolution (CGME).
      </p>
    </PageSection>

    <section class="section-band teaser-band">
      <div class="section-inner teaser-inner">
        <FigurePanel
          :src="assetUrl('fig/fig1b.png')"
          alt="Distribution of failure modes under the Einstellung Effect across representative reasoning LLMs"
          caption="Distribution of failure modes under the Einstellung Effect across representative reasoning LLMs."
        />
      </div>
    </section>

    <PageSection id="motivation" title="Motivation" tone="soft">
      <p>
        Medical LLMs can score well on conventional benchmarks while still
        failing when atypical evidence contradicts familiar disease patterns. In
        differential diagnosis, the critical test is whether a model follows
        patient-specific discriminative evidence instead of statistical
        shortcuts. MedEinst makes this Einstellung Effect measurable through
        counterfactual control-trap case pairs.
      </p>
    </PageSection>

    <PageSection
      id="dataset-pipeline"
      title="Benchmark Construction / Dataset Pipeline"
    >
      <FigurePanel
        :src="assetUrl('fig/Fig2.png')"
        alt="MedEinst benchmark construction pipeline figure"
        caption="MedEinst constructs counterfactual control-trap pairs by modifying discriminative evidence while preserving most clinical context."
      />
    </PageSection>

    <PageSection
      id="benchmark"
      title="Benchmark Overview"
      lede="Counterfactual differential diagnosis for measuring the Einstellung Effect."
    >
      <StatCards :stats="stats" />
      <p>
        Each MedEinst pair keeps most clinical context stable while modifying
        discriminative evidence so that the correct diagnosis changes. This
        control-trap setup makes it possible to separate broad diagnostic
        accuracy from susceptibility to the Einstellung Effect.
      </p>
    </PageSection>

    <PageSection
      id="failure-modes"
      title="Failure Patterns"
      tone="soft"
    >
      <FailureModePie />
    </PageSection>

    <PageSection id="ecr-agent" title="ECR-Agent Overview" tone="soft">
      <FigurePanel
        :src="assetUrl('fig/Fig3.png')"
        alt="ECR-Agent architecture overview figure"
        caption="ECR-Agent aligns diagnosis with explicit evidence review, causal reasoning, and memory-guided graph evolution."
      />
      <div class="explanation-grid" aria-label="ECR-Agent components">
        <article v-for="card in ecrCards" :key="card.title" class="explanation-card">
          <h3>{{ card.title }}</h3>
          <p>{{ card.body }}</p>
        </article>
      </div>
    </PageSection>

    <PageSection
      id="performance"
      title="Overall Performance"
      lede="Overall Performance on MedEinst"
    >
      <PerformanceTable />
    </PageSection>

    <PageSection
      id="interactive-results"
      title="Interactive Results"
      tone="soft"
    >
      <InteractiveFrame
        :src="assetUrl('html/ACL_Final_Figures_Clean.html')"
        title="Interactive MedEinst Plotly results for Fig. 4-5"
        height="min(1040px, max(560px, 80vw))"
      />
    </PageSection>

    <PageSection id="citation" title="Citation">
      <CitationBlock />
    </PageSection>

    <SiteFooter />
  </div>
</template>
