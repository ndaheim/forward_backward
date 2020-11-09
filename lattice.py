import numpy as np

from semiring import LogSemiring, ProbabilitySemiring

class Node(object):

    def __init__(self, I, t, v=None, W=None, in_links=None, out_links=None):
        if out_links is None:
            out_links = []
        if in_links is None:
            in_links = []
        self.index = I
        self.time = t
        self.var = v
        self.word = W
        self.in_links = in_links
        self.out_links = out_links
        self.best_predecessor = None

class Link(object):

    def __init__(self, J, S, E, W, v, a, l):
        self.acoustic_score = a
        self.lm_score = l
        self.index = J
        self.start = S
        self.end = E
        self.word = W
        self.var = v
        self.forward_backward_score = None

    def score(self, lm_scale):
        return (lm_scale * self.lm_score) + self.acoustic_score


class Lattice(object):
    """
    Implements a lattice represention recognized hypotheses in ASR.

    Example:
        >>> from semiring import ProbabilitySemiring

        >>> from util import read_htk

        >>> lattice = read_htk(htk_path)

        >>> scored_lattice = lattice.forward_backward(ProbabilitySemiring())
    """

    def __init__(self, nodes=None, links=None, lm_scale=1.0, version=None, utterance=None):
        if nodes is None:
            nodes = []
        if links is None:
            links = []
        self.nodes = nodes
        self.links = links
        self.lm_scale = lm_scale
        self.version = version
        self.utterance = utterance

    def get_node_by_index(self, index):
        for node in self.nodes:
            if node.index == index:
                return node
        return None

    @property
    def end_node(self):
        return self.nodes[-1]

    @property
    def start_node(self):
        return self.nodes[0]

    def edge_posterior(self, lattice, semiring):
        """
        Augments each link in the lattice by its edge posterior.
        
        Args:
            lattice: Scored lattice representing recognized hypotheses.
            semiring: LogSemiring or ProbabilitySemiring implementing addition and multiplication.
        """
        if not any([isinstance(semiring, semiring_class) for semiring_class in [LogSemiring, ProbabilitySemiring]]):
            raise Exception(f"Edge posterior not implemented in the semiring {type(semiring)}")

        p_X = lattice.end_node.forward

        for link in lattice.links:
            
            link.posterior = semiring.multiply([
                link.start.forward,
                link.score(lattice.lm_scale),
                link.end.backward,
                -p_X if isinstance(semiring, LogSemiring) else 1/p_X
            ])

    def forward_backward(self, semiring):
        """
        Implements the forward backward algorithm under the given semiring.

        Args:
            semiring: Semiring implementing addition, multiplication as well as
                      the zero elements w.r.t to both operations.
        Returns:
            Topologically sorted lattice.
        """
        reverse_sort = self.topological_sort(0)
        sort = reverse_sort[::-1]

        self.start_node.forward = semiring.one()
        for node_id in sort[1:]:
            node = self.nodes[node_id]
            node.forward = semiring.zero()
            for link in node.in_links:
                node.forward = semiring.add([
                    node.forward,
                    semiring.multiply([link.start.forward, link.score(self.lm_scale)])
                ])

        self.end_node.backward = semiring.one()
        for node_id in reverse_sort[1:]:
            node = self.nodes[node_id]
            node.backward = semiring.zero()
            for link in node.out_links:
                node.backward = semiring.add([
                    node.backward,
                    semiring.multiply([link.end.backward, link.score(self.lm_scale)])
                ])

        # assert correctness of the result within numerical bounds
        np.testing.assert_allclose(
            [self.start_node.backward],
            [self.end_node.forward]
        )

        return reverse_sort

    def rescore(self, semiring):
        """Rescores the lattie based on the edge posteriors."""
        for link in self.links:
            duration = (link.end.time - link.start.time) / .01
            time_frames = [(link.start.time + time_frame * .01)
                        for time_frame in range(0, round(duration))]

            # calculate cost as 1 - 1/duration * sum(time frame word posteriors)
            link_posterior = semiring.add([
                self.time_frame_word_posterior(semiring, link.word, time_frame)
                for time_frame in time_frames]
            )
            time_frames = -np.log(len(time_frames)) if isinstance(semiring, LogSemiring) else 1/len(time_frames)
            link.rescored = semiring.add([
                semiring.one(),
                -semiring.multiply([link_posterior, time_frames])
            ])

    def time_frame_word_posterior(self, semiring, word, time):
        """Returns the time frame word posterior."""
        time_frame_posterior = semiring.add([
            link.posterior for link in self.links
            if link.word == word and link.start.time <= time < link.end.time])

        return time_frame_posterior

    def topological_sort(self, root):
        """Sorts the lattice topologically from the given root on."""
        root = self.nodes[root]
        visited = []
        reverse_sort = []

        def rec_visit(node):
            node_id = node.index
            if node_id in visited:
                return
            visited.append(node_id)
            for link in node.out_links:
                rec_visit(link.end)
            reverse_sort.append(node_id)

        rec_visit(root)
        return reverse_sort