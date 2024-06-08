import torch
from torch.utils.data import DataLoader
from misc.data_fetch import DatasetSplit


class LocalTrainer_HN(object):
    """User-side trainer: only pass information that are needed"""
    def __init__(self, dataset_tr, idxs_tr, local_bs):
        self.loss_func = torch.nn.CrossEntropyLoss()
        tmp_dataset_tr = DatasetSplit(dataset_tr, idxs_tr)
        self.local_tr_sample_ct = len(tmp_dataset_tr)
        self.ldr_train = DataLoader(dataset=tmp_dataset_tr,
                                    batch_size=local_bs,
                                    shuffle=True)
        del tmp_dataset_tr

    def do_train(self, net, lr, momentum, local_ep, grad_clip, device):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        local_epoch_loss = []
        sample_ct = 0

        for local_iter in range(local_ep):
            batch_loss = []

            for bidx, (images, labels) in enumerate(self.ldr_train):
                # clear grad
                optimizer.zero_grad()
                # model feedforward
                images, labels = images.to(device), labels.to(device)
                log_probs = net(images)
                # compute cross-entropy loss
                local_loss = self.loss_func(log_probs, labels)
                if torch.isnan(local_loss):
                    print("Loss is nan. Should check if there is an issue.")
                # backpropagate
                local_loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                # weight update
                optimizer.step()
                # record keeping
                batch_loss.append(local_loss.item())
                sample_ct += images.size(0)
            # compute average loss for every iter
            local_epoch_loss.append(sum(batch_loss) / len(batch_loss))
        w_net = net.state_dict()
        avg_loss = sum(local_epoch_loss) / len(local_epoch_loss)
        return w_net, avg_loss, sample_ct


class LocalTester_HN(object):
    """User-side tester: only pass information that are needed"""

    def __init__(self, dataset_te, idxs_te, local_bs):
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="sum")
        tmp_dataset_te = DatasetSplit(dataset_te, idxs_te)
        self.local_te_sample_ct = len(tmp_dataset_te)
        self.ldr_test = DataLoader(dataset=tmp_dataset_te,
                                   batch_size=local_bs,
                                   shuffle=False)
        del tmp_dataset_te

    def do_test(self, net, device):
        net.eval()
        with torch.no_grad():
            test_loss = 0
            correct = 0
            for bidx, (images, labels) in enumerate(self.ldr_test):
                # model feedforward
                images, labels = images.to(device), labels.to(device)
                log_probs = net(images)
                # sum up batch loss
                test_loss += self.loss_func(log_probs, labels).item()
                # get the index of the max log-probability
                y_pred = log_probs.data.max(1, keepdim=True)[1]  # get highest score
                correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
        test_loss /= self.local_te_sample_ct  # total loss divide by generate count
        accuracy = 100.00 * float(correct) / self.local_te_sample_ct  # total acc divide by generate count
        return accuracy, test_loss