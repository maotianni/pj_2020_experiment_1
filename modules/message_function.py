from torch import nn

class MessageFunction(nn.Module):
    """
    Module which computes the message for a given interaction.
    """

    def compute_message(self, raw_messages):
        return None


class IdentityMessageFunction(MessageFunction):
    def compute_message(self, raw_messages):
        return raw_messages


def get_message_function(module_type, raw_message_dimension, message_dimension):
    if module_type == "identity":
        return IdentityMessageFunction()
    else:
        raise ValueError("Message function {} not implemented".format(module_type))


