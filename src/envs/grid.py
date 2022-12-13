import torch


class Grid:
    def __init__(self, x1, x2, y1, y2, x_ticks, y_ticks, omitted_shape) -> None:
        self.x1, self.x2 = x1, x2
        self.y1, self.y2 = y1, y2
        self.x_ticks, self.y_ticks = x_ticks, y_ticks
        self.omitted_shape = omitted_shape
        self.adj_list = self._initialize_graph()

    def _initialize_graph(self):
        x = torch.linspace(self.x1, self.x2, self.x_ticks)
        y = torch.linspace(self.y1, self.y2, self.y_ticks)
        xx, yy = torch.meshgrid(x, y)
        adj_list = {}

        for i in range(len(x)):
            for j in range(len(y)):
                if self.omitted_shape == None or (not self.omitted_shape.overlaps([xx[i, j], yy[i, j]])):
                    neighbors = []
                    if i - 1 >= 0 and (
                        self.omitted_shape == None or not self.omitted_shape.overlaps([xx[i - 1, j], yy[i - 1, j]])
                    ):
                        neighbors.append(torch.tensor([xx[i - 1, j], yy[i - 1, j]]))
                    if i + 1 < len(x) and (
                        self.omitted_shape == None or not self.omitted_shape.overlaps([xx[i + 1, j], yy[i + 1, j]])
                    ):
                        neighbors.append(torch.tensor([xx[i + 1, j], yy[i + 1, j]]))
                    if j - 1 >= 0 and (
                        self.omitted_shape == None or not self.omitted_shape.overlaps([xx[i, j - 1], yy[i, j - 1]])
                    ):
                        neighbors.append(torch.tensor([xx[i, j - 1], yy[i, j - 1]]))
                    if j + 1 < len(y) and (
                        self.omitted_shape == None or not self.omitted_shape.overlaps([xx[i, j + 1], yy[i, j + 1]])
                    ):
                        neighbors.append(torch.tensor([xx[i, j + 1], yy[i, j + 1]]))
                    adj_list[(xx[i, j], yy[i, j])] = neighbors

        return adj_list

    def sample_random_state(self):
        state = list(self.adj_list.keys())
        idx = torch.randint(len(state), (1,)).item()

        return torch.tensor(state[idx])
