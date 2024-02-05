import torch
from gym import Wrapper
import wandb


class DiscrRewards(Wrapper):
    def __init__(self, env, discriminator, device, online_train, num_frames, num_envs, pretrain):
        super().__init__(env)
        self.device = device

        self.discriminator = discriminator.to(device)

        self.online_train = online_train
        self.num_frames = num_frames

        self.total_data = 0

        self.num_envs = num_envs
        self.env_ids = torch.arange(self.num_envs, requires_grad=False, device=self.device)

        # if self.online_train:
        #     root_path = "./discr_data/"
        #     expertloader = dataloader(folder_path=root_path)
        #     train_loader, _ = expertloader.createloader()
        #     self.loader = iter(train_loader)

        from src.utils.dataset import get_dataset
        self.dataset = get_dataset()

        if pretrain:
            print("Pretraining discr...")
            from tqdm import tqdm
            for _ in tqdm(range(1000)):
                self.update()
            print("Done.")

        print("Proceed without worries!")

    def update(self, online_data=None):
        if online_data is None:
            batch_size = 3600
        else:
            batch_size = self.num_envs

        # gail-style

        offline_input, offline_labels = self.dataset.get_batch(batch_size)
        offlne_labels = torch.concat([torch.zeros((batch_size, 3), device='cuda'), offline_labels], dim=1)

        if online_data is not None:
            online_input = online_data[:, :, :-3].unsqueeze(dim=1).to(self.device)
            online_labels = online_data[:, 49, -3:].to(self.device)

            online_labels = torch.concat([online_labels, torch.zeros((batch_size, 3), device='cuda')], dim=1)

            all_labels = torch.concat([offlne_labels, online_labels], dim=0)
            all_input = torch.concat([offline_input, online_input], dim=0)
        else:
            all_labels = offlne_labels
            all_input = offline_input

        self.discriminator.optimizer.zero_grad()
        outputs = self.discriminator.forward(all_input)

        loss = self.discriminator.loss_criterion(outputs, all_labels)
        loss.backward()
        self.discriminator.optimizer.step()

        train_acc = torch.eq(torch.argmax(outputs, dim=1), torch.argmax(all_labels, dim=1)).float().mean()

        if online_data is not None:
            wandb.log({
                "charts/training_loss": loss.detach().item(),
                "charts/training_accuracay": train_acc,
            })

    def step(self, action):

        obs, rew, done, info = self.env.step(action)

        assert info['framestack'] is not None
        stacked_obs = info['framestack']

        # if self.online_train:
            
        #     # import pdb; pdb.set_trace()
        #     batch = next((self.loader))    

        #     if batch.shape[0] == 50:

        #         batch = batch.movedim(0,1) #Due to how the dataloader works
        #         batch = batch[~batch.isinf().any(dim=1).any(dim=1),:,:]

        #         labels_exp = batch[:,0,-3:].to(self.device)
        #         inputs_exp = batch[:,:,:-3].unsqueeze(dim=1).to(self.device)

        #         labels = torch.vstack((stacked_obs[:,0,-3:].to(self.device),labels_exp))
        #         inputs = torch.vstack((stacked_obs[:,:,:-3].unsqueeze(dim=1).to(self.device),inputs_exp))

        #     else:
        #         labels = stacked_obs[:,0,-3:].to(self.device)
        #         inputs = stacked_obs[:,:,:-3].unsqueeze(dim=1).to(self.device)

        #     self.discriminator.optimizer.zero_grad()
        #     outputs = self.discriminator.forward(inputs)

        #     loss = self.discriminator.loss_criterion(outputs, labels)
        #     loss.backward()
        #     self.discriminator.optimizer.step()


        #     self.total_data += labels.shape[0]
        #     train_acc = torch.eq(torch.argmax(outputs, dim=1),torch.argmax(labels,dim=1)).float().mean()

        #     wandb.log({
        #         "charts/train_data_used": self.total_data,
        #         "charts/training_loss": loss.item(),
        #         "charts/training_accuracay": train_acc,
        #                 })


        if self.online_train:
            self.update(stacked_obs)

        with torch.no_grad():
            rew_disc = self.discriminator(stacked_obs[:,:,:-3].unsqueeze(dim=1))
            correct_ids = stacked_obs[:,-1,-3:]

            #penalty = -rew_disc.sum(dim=1)

            correct_ids = correct_ids.argmax(dim=1)
            if self.online_train:
                correct_ids += 3 # actually, we want to look at the offline preds, not online preds

            rew_disc_for_ids = rew_disc[self.env_ids,correct_ids]
            #rew_disc_for_ids += penalty

            # rew is 1 * rew for id - 1 * rew for other ids
            # incentivizes big differences in rews

        if not self.online_train:
            rew += rew_disc_for_ids.detach() * 10
        else:
            rew = rew_disc_for_ids.detach() * 10

        return obs, rew, done, info













