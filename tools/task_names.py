"""未来云端表单的任务名约定；不改变历史任务ID、进程或模型合同。"""
import argparse
PREFIX='zhounan_'
DEFAULT_NAMES={
    'annotation':'zhounan_redimnet2_annotation_8a800_p6',
    'ptn_resume':'zhounan_redimnet2_ptn_resume_8a800_p6',
    'word_finetune':'zhounan_redimnet2_word_finetune_8a800_p6',
}


def validate_task_name(value):
    """只实施用户的前缀约束；不假造平台的长度或字符集规则。"""
    if (not isinstance(value,str) or not value.startswith(PREFIX) or len(value)<=len(PREFIX)
            or value!=value.strip() or any(ord(c)<32 or ord(c)==127 for c in value)):
        raise ValueError('未来云任务名必须以zhounan_开头，后缀非空且不能含首尾空白或控制字符')
    return value


def suggested_task_name(workload,value=None):
    if workload not in DEFAULT_NAMES:raise ValueError('unknown future cloud workload')
    return validate_task_name(DEFAULT_NAMES[workload]if value is None else value)


def task_name_argument(value):
    """在CLI中保留具体中文原因，而不让argparse吞掉ValueError说明。"""
    try:return validate_task_name(value)
    except ValueError as exc:raise argparse.ArgumentTypeError(str(exc))from exc
