# APT Distillation for Image Generation

## Description

Этот репозиторий содержит код для Adversarial Post-Training (APT) Distillation — метода, описанного в статье Adversarial Post-Training. Мы применяем APT Distillation для улучшения дистиллированных диффузионных моделей, начиная с Consistency Distillation, а затем обучая APT-Discriminator для генерации реалистичных изображений.

Процесс обучения проходит в два этапа:

1. Consistency Distillation

2. APT Distillation (Adversarial Fine-Tuning)

Используется модель micro-dit
