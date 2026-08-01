import torch
import torch.nn.functional as F
from torch import nn

def build_loss(): return nn.BCEWithLogitsLoss()


def kd_logits_loss(student_logits, teacher_logits, teacher_slice=None, reduction: str = "mean"):
    """Softmax logits distillation KL (Hinton-style, temperature 1).

    KL(log_softmax(student) || log_softmax(teacher)), summed over classes and
    divided by batch size.  log_target=True keeps it numerically stable and
    lets teacher logits be pre-normalized.  Explicit .float() keeps the op
    fp32 under bf16 autocast.

    `teacher_slice` (LongTensor of teacher indices in student tag order)
    supports KD onto a student with a subset of the teacher's vocabulary: the
    teacher log-softmax is computed over its FULL distribution, then sliced to
    the student's tags.  None = identical vocab (current path).

    Returns the mean over the batch (reduction="mean"), or the raw summed
    value (reduction="sum").
    """
    student = student_logits.float().log_softmax(-1)
    teacher = teacher_logits.float().log_softmax(-1)
    if teacher_slice is not None:
        teacher = teacher[:, teacher_slice]
    kld = F.kl_div(student, teacher, reduction="sum", log_target=True)
    if reduction == "sum":
        return kld
    return kld / student_logits.size(0)
