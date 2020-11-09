import csv
import os
import subprocess

from lattice import Lattice, Link, Node

def read_htk(file_name):
    nodes = []
    links = []

    with open(file_name, "r") as f:
        lattice = f.read()
        num_nodes = int(lattice.split("NODES=")[1].split()[0])
        num_links = int(lattice.split("LINKS=")[1].split()[0])
        version = lattice.split("VERSION=")[1].split()[0]
        utterance = lattice.split("UTTERANCE=")[1].split()[0]
        
        for line in lattice.split("\n"):
            line = line.strip()
            if line.startswith("lmscale"):
                lm_scale = float(line.split("=")[1])
            if line.startswith("I="):
                tokens = line.split(" ")
                nodes.append(Node(
                        int(tokens[0].split("=")[1]),
                        float(tokens[1].split("=")[1])
                ))
            if line.startswith("J="):
                tokens = line.split(" ")
                start = nodes[int(tokens[1].split("=")[1])]
                end = nodes[int(tokens[2].split("=")[1])]
                links.append(
                    Link(
                        int(tokens[0].split("=")[1]),
                        start,
                        end,
                        tokens[3].split("=")[1].strip("\""),
                        int(tokens[4].split("=")[1]),
                        float(tokens[5].split("=")[1]),
                        float(tokens[6].split("=")[1])
                    )
                )
        for link in links:
            link.start.out_links.append(link)
            link.end.in_links.append(link)

        print("Read lattice with {} nodes, {} links and lm scale {}".format(
            num_nodes,
            num_links,
            lm_scale
        ))
        assert len(nodes) == num_nodes
        assert len(links) == num_links

        return Lattice(
            nodes=nodes,
            links=links,
            lm_scale=lm_scale,
            utterance=utterance,
            version=version
        )

class CTMWriter(object):
    """Implements a writer for lattices to .ctm files."""

    HEADER = [";;", "<name>", "<track>", "<start>", "<duration>", "<word>"]

    def _end_time(self, utterance):
        split_ = utterance.split("_")
        end = split_[-1].lstrip("0")
        end = end[:-3] + "." + end[-3:]
        return "{:.3f}".format(round(float(end), 3))

    def _start_time(self, utterance):
        split_ = utterance.split("_")
        start = split_[-2].lstrip("0")
        start = start[:-3] + "." + start[-3:]
        return "{:.3f}".format(round(float(start), 3))

    def _utterance_name(self, utterance):
        split_ = utterance.split("_")
        return "_".join(split_[:-2])

    def description(self, lattice):
        row = [";;"]
        utterance = self._utterance_name(lattice.utterance)
        start = self._start_time(lattice.utterance)
        end = self._end_time(lattice.utterance)
        full_utterance_desc = utterance + "/" + lattice.utterance
        row.append(full_utterance_desc)
        start_end = "(" + start + "-" + end + ")"
        row.append(start_end)
        return row

    def write_ctm(self, hypotheses, lattices, file_name):
        with open(file_name, "w", encoding="utf-8", newline="") as ctm_file:
            writer = csv.writer(ctm_file, delimiter=" ", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
            
            for hypothesis, lattice in zip(hypotheses, lattices):
                writer.writerow(self.HEADER)
                writer.writerow(self.description(lattice))
                utterance = self._utterance_name(lattice.utterance)
                start_time = self._start_time(lattice.utterance)
                
                for link in hypothesis:
                    start = "{:.3f}".format(round((link.start.time + float(start_time)), 3))
                    
                    if link.word not in ["[SILENCE]", "[NOISE]", "!NULL"]:
                        row = [utterance, "1", start]
                        duration = link.end.time - link.start.time
                        row.append("{:.3f}".format(round(duration, 3)))
                        row.append(link.word.rstrip())
                        writer.writerow(row)
        return

def wer(file_name_hyp, file_name_ref, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    try:
        with subprocess.Popen(
                [
                    "./sclite",
                    "-e", "utf-8",
                    "-r", file_name_ref,
                    "stm",
                    "-h", file_name_hyp,
                    "ctm",
                    "-o", "all",
                    "-O", output_folder,
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:

            for line in p.stdout:
                line = str(line)[2:-1].replace("\\n", "").replace("\\t", "   ")
                print(line, end='\n')
            for line in p.stderr:
                line = str(line)[2:-1].replace("\\n", "").replace("\\t", "   ")
                print(line, end='\n')

    except subprocess.CalledProcessError as e:
        print(e.returncode)
        print(e.output)

    path = os.path.join(output_folder, file_name_hyp+".sys")
    with open(path, "r") as f:
        return f.read().split("Sum/Avg")[1].split()[8]