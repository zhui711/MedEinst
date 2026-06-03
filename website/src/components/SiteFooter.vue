<script setup>
import { links } from '@/data/project'
</script>

<template>
  <footer class="site-footer">
    <div class="footer-inner">
      <p>Contact: <a href="mailto:zhongruizhu@stu.xjtu.edu.cn">zhongruizhu@stu.xjtu.edu.cn</a></p>
      <nav aria-label="Footer project links">
        <a :href="links.paper" target="_blank" rel="noreferrer">Paper</a>
        <a :href="links.code" target="_blank" rel="noreferrer">Code</a>
        <a :href="links.dataset" target="_blank" rel="noreferrer">Dataset</a>
        <a :href="links.slides" target="_blank" rel="noreferrer">Slides</a>
      </nav>
    </div>
  </footer>
</template>
