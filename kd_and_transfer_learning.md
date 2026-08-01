# Motivation

- I have a very strong SOTA DINOv3 L/16 model, and I want to scale down to
smaller models (i.e., DINOv3 B/16, S/16).
- Transfer learning & KD are two of the most straightforward ways

# Transfer Learning
- The head is very wise due to learning from 1m images and 11k tags, the idea
is to transfer its weights
- Variants of DINOv3 has different token dims, so we need another projection
layer for the student model if we want to keep the head the same as the teacher model:
  - add another Projector class in @src/model.py, a simple nn.Linear
  - only enable it if the number of dims from --checkpoint (@src/train.py) is
  different from the current model
  - add `head_lr_mult` and `proj_lr_mult` to @src/config.py, we can set 0.01x for the head for example
  - add `projector = new dims` to @src/config.py so we can know how to load the checkpoint
    after training with a projector after training
  - projector is completely optional so please support both models with and
  without it
  - Projector participates in:
    - optimizer
    - scheduler
    - EMA

# KD
- KD should be simpler than transfer learning, logits distillation is enough:
  loss = BCE(gt, student_logits) + kd_weight * KL(student_logits, teacher_logits)
  (--kd-weight passed alongside --teacher-path)
- We can use .onnx or .pt for this, add --teacher-path to @src/train.py. For
  .pt the model must be in eval mode and frozen.
