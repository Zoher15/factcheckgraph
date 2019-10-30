fcgclass=['fred','co-occur','backbone']
fcglabels={'fred':['tfcg','ffcg','ufcg'],'co-occur':['tfcg_co','ffcg_co','ufcg_co'],'backbone':['tfcg_bb','ffcg_bb','ufcg_bb']}
kgtype=['dbpedia','wikidata']
modes=["all","adj","nonadj"]
graphpath="/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/graphs"
rdfpath="/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/rdf_files"
analyzepath="/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph/analyze_graphs"
createpath="/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph/create_graphs"
processpath="/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph/process_graphs"
plotpath="/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph/plot_graphs"
rule plot_graphs:
    input:
        "{graphpath}/{fcgclass}/intersect_adj_{fcgclass}_{kgtype}_ID.json",
        "{graphpath}/{fcgclass}/intersect_adj_{kgtype}_{fcglabels[fcgclass][0]}_ID.json"
        "{graphpath}/{fcgclass}/intersect_nonadj_{kgtype}_{fcglabels[fcgclass][0]}_ID.json"
        "{graphpath}/{fcgclass}/intersect_adj_{kgtype}_{fcglabels[fcgclass][1]}_ID.json"
        "{graphpath}/{fcgclass}/intersect_nonadj_{kgtype}_{fcglabels[fcgclass][1]}_ID.json"
        "{graphpath}/{fcgclass}/intersect_adj_{kgtype}_{fcglabels[fcgclass][2]}_ID.json"
        "{graphpath}/{fcgclass}/intersect_nonadj_{kgtype}_{fcglabels[fcgclass][2]}_ID.json"
    output:
        "/gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph_data/plots/adj_pairs/true(adj)_vs_false(non-adj)_pairs_{kgtype}_{fcgclass}.png"
    shell:
        "python {plotpath}/plot_adj_pairs.py -fcg {fcgclass} -kg {kgtype}"

rule find_adj_pairs:
    input:
        "{graphpath}/{fcgclass}/{fcglabels[fcgclass][i]}/data/intersect_all_entityPairs_{kgtype}_{fcglabels[fcgclass][i]}_IDs.json"
    output:
        "{graphpath}/{fcgclass}/{fcglabels[fcgclass][i]}/data/intersect_adj_entityPairs_{kgtype}_{fcglabels[fcgclass][i]}_IDs.json"
        "{graphpath}/{fcgclass}/{fcglabels[fcgclass][i]}/data/intersect_nonadj_entityPairs_{kgtype}_{fcglabels[fcgclass][i]}_IDs.json"
    shell:
        "python {analyzepath}/find_adj_pairs.py -fcg {fcgclass} -kg {kgtype}"

rule fcg_klinker:
	envs:
		"env-klinker"
	input:
		"{graphpath}/{fcgclass}/{fcglabels[fcgclass][i]}/data/{fcglabels[fcgclass][i]}_nodes.txt",
		"{graphpath}/{fcgclass}/{fcglabels[fcgclass][i]}/data/{fcglabels[fcgclass][i]}_edgelistID.npy",
		"{graphpath}/{fcgclass}/{fcglabels[fcgclass][i]}/data/intersect_all_entityPairs_{kgtype}_{fcglabels[fcgclass][i]}_IDs_klformat.txt",
	output:
		"{graphpath}/{fcgclass}/{fcglabels[fcgclass][i]}/data/intersect_all_entityPairs_{kgtype}_{fcglabels[fcgclass][i]}_IDs.json"
	shell:
		"klinker linkpred {graphpath}/{fcgclass}/{fcglabels[fcgclass][i]}/data/{fcglabels[fcgclass][i]}_nodes.txt {graphpath}/{fcgclass}/{fcglabels[fcgclass][i]}/data/{fcglabels[fcgclass][i]}_edgelistID.npy {graphpath}/{fcgclass}/{fcglabels[fcgclass][i]}/data/intersect_all_entityPairs_{fcglabels[fcgclass][i]}_IDs_klformat.txt {graphpath}/{fcgclass}/{fcglabels[fcgclass][i]}/data/intersect_all_entityPairs_{fcglabels[fcgclass][i]}_IDs.json -u -n 12 -w logdegree"

rule kg_klinker:
	envs:
		"env-klinker"
	input:
		"{graphpath}/kg/{kgtype}/data/{kgtype}_nodes.txt",
		"{graphpath}/kg/{kgtype}/data/{kgtype}_edgelistID.npy",
		"{graphpath}/kg/{kgtype}/data/intersect_all_entityPairs_{fcgclass}_{kgtype}_IDs_klformat.txt",
	output:
		"{graphpath}/kg/{kgtype}/data/intersect_all_entityPairs_{fcgclass}_{kgtype}_IDs.json"
	shell:
		"klinker linkpred {graphpath}/kg/{kgtype}/data/{kgtype}_nodes.txt {graphpath}/kg/{kgtype}/data/{kgtype}_edgelistID.npy {graphpath}/kg/{kgtype}/data/intersect_all_entityPairs_{fcglabels[fcgclass][i]}_IDs_klformat.txt {graphpath}/kg/{kgtype}/data/intersect_all_entityPairs_{fcglabels[fcgclass][i]}_IDs.json -u -n 12 -w logdegree"

rule create_bbdc_ufcg:
	input:
		"{graphpath}/co-occur/intersect_entities_dbpedia_co-occur.txt",
		"{graphpath}/co-occur/ufcg/ufcg.edgelist"
	output:
		"{graphpath}/backbone_dc/ufcg_bbdc/ufcg_bbdc.edgelist",
		"{graphpath}/backbone_dc/ufcg_bbdc/ufcg_bbdc.graphml"
		"{graphpath}/backbone_dc/ufcg_bbdc/data/ufcg_bbdc_entities.txt",
		"{graphpath}/backbone_dc/ufcg_bbdc/data/ufcg_bbdc_node2ID.json",
		"{graphpath}/backbone_dc/ufcg_bbdc/data/ufcg_bbdc_edgelistID.npy"
	shell:
		"python {createpath}/create_backbone.py -fcg co-occur -ft ufcg -kg dbpedia"

rule create_bbdc_ffcg:
	input:
		"{graphpath}/co-occur/intersect_entities_dbpedia_co-occur.txt",
		"{graphpath}/co-occur/ffcg/ffcg.edgelist"
	output:
		"{graphpath}/backbone_dc/ffcg_bbdc/ffcg_bbdc.edgelist",
		"{graphpath}/backbone_dc/ffcg_bbdc/ffcg_bbdc.graphml"
		"{graphpath}/backbone_dc/ffcg_bbdc/data/ffcg_bbdc_entities.txt",
		"{graphpath}/backbone_dc/ffcg_bbdc/data/ffcg_bbdc_node2ID.json",
		"{graphpath}/backbone_dc/ffcg_bbdc/data/ffcg_bbdc_edgelistID.npy"
	shell:
		"python {createpath}/create_backbone.py -fcg co-occur -ft ffcg -kg dbpedia"

rule create_bbdc_tfcg:
	input:
		"{graphpath}/co-occur/intersect_entities_dbpedia_co-occur.txt",
		"{graphpath}/co-occur/tfcg/tfcg.edgelist"
	output:
		"{graphpath}/backbone_dc/tfcg_bbdc/tfcg_bbdc.edgelist",
		"{graphpath}/backbone_dc/tfcg_bbdc/tfcg_bbdc.graphml"
		"{graphpath}/backbone_dc/tfcg_bbdc/data/tfcg_bbdc_entities.txt",
		"{graphpath}/backbone_dc/tfcg_bbdc/data/tfcg_bbdc_node2ID.json",
		"{graphpath}/backbone_dc/tfcg_bbdc/data/tfcg_bbdc_edgelistID.npy"
	shell:
		"python {createpath}/create_backbone.py -fcg co-occur -ft tfcg -kg dbpedia"

rule create_bbdf_ufcg:
	input:
		"{graphpath}/fred/intersect_entities_dbpedia_fred.txt",
		"{graphpath}/fred/ufcg/ufcg.edgelist"
	output:
		"{graphpath}/backbone_df/ufcg_bbdf/ufcg_bbdf.edgelist",
		"{graphpath}/backbone_df/ufcg_bbdf/ufcg_bbdf.graphml"
		"{graphpath}/backbone_df/ufcg_bbdf/data/ufcg_bbdf_entities.txt",
		"{graphpath}/backbone_df/ufcg_bbdf/data/ufcg_bbdf_node2ID.json",
		"{graphpath}/backbone_df/ufcg_bbdf/data/ufcg_bbdf_edgelistID.npy"
	shell:
		"python {createpath}/create_backbone.py -fcg fred -ft ufcg -kg dbpedia"

rule create_bbdf_ffcg:
	input:
		"{graphpath}/fred/intersect_entities_dbpedia_fred.txt",
		"{graphpath}/fred/ffcg/ffcg.edgelist"
	output:
		"{graphpath}/backbone_df/ffcg_bbdf/ffcg_bbdf.edgelist",
		"{graphpath}/backbone_df/ffcg_bbdf/ffcg_bbdf.graphml"
		"{graphpath}/backbone_df/ffcg_bbdf/data/ffcg_bbdf_entities.txt",
		"{graphpath}/backbone_df/ffcg_bbdf/data/ffcg_bbdf_node2ID.json",
		"{graphpath}/backbone_df/ffcg_bbdf/data/ffcg_bbdf_edgelistID.npy"
	shell:
		"python {createpath}/create_backbone.py -fcg fred -ft ffcg -kg dbpedia"

rule create_bbdf_tfcg:
	input:
		"{graphpath}/fred/intersect_entities_dbpedia_fred.txt",
		"{graphpath}/fred/tfcg/tfcg.edgelist"
	output:
		"{graphpath}/backbone_df/tfcg_bbdf/tfcg_bbdf.edgelist",
		"{graphpath}/backbone_df/tfcg_bbdf/tfcg_bbdf.graphml"
		"{graphpath}/backbone_df/tfcg_bbdf/data/tfcg_bbdf_entities.txt",
		"{graphpath}/backbone_df/tfcg_bbdf/data/tfcg_bbdf_node2ID.json",
		"{graphpath}/backbone_df/tfcg_bbdf/data/tfcg_bbdf_edgelistID.npy"
	shell:
		"python {createpath}/create_backbone.py -fcg fred -ft tfcg -kg dbpedia"

rule find_intersect_co:
	input:
		"{graphpath}/co-occur/tfcg_co/data/tfcg_co_entities.txt",
		"{graphpath}/co-occur/ffcg_co/data/ffcg_co_entities.txt",
		"{graphpath}/kg/{kgtype}/data/{kgtype}_entities.txt",
	output:
		"{graphpath}/kg/{kgtype}/data/intersect_all_entityPairs_{kgtype}_co-occur_{kgtype}_IDs_klformat.txt",
		"{graphpath}/co-occur/intersect_entities_{kgtype}_co-occur.txt",
		"{graphpath}/co-occur/tfcg_co/data/intersect_all_entityPairs_{kgtype}_co-occur_tfcg_co_IDs_klformat.txt",
		"{graphpath}/co-occur/ffcg_co/data/intersect_all_entityPairs_{kgtype}_co-occur_ffcg_co_IDs_klformat.txt",
		"{graphpath}/co-occur/ufcg_co/data/intersect_all_entityPairs_{kgtype}_co-occur_ufcg_co_IDs_klformat.txt",
	shell:
		"python {processpath}/find_intersect.py -fcg co-occur -kg {kgtype}"

rule find_intersect_fred:
	input:
		"{graphpath}/fred/tfcg/data/tfcg_entities.txt",
		"{graphpath}/fred/ffcg/data/ffcg_entities.txt",
		"{graphpath}/kg/{kgtype}/data/{kgtype}_entities.txt",
	output:
		"{graphpath}/kg/{kgtype}/data/intersect_all_entityPairs_{kgtype}_fred_{kgtype}_IDs_klformat.txt",
		"{graphpath}/fred/intersect_entities_{kgtype}_fred.txt",
		"{graphpath}/fred/tfcg/data/intersect_all_entityPairs_{kgtype}_fred_tfcg_IDs_klformat.txt",
		"{graphpath}/fred/ffcg/data/intersect_all_entityPairs_{kgtype}_fred_ffcg_IDs_klformat.txt",
		"{graphpath}/fred/ufcg/data/intersect_all_entityPairs_{kgtype}_fred_ufcg_IDs_klformat.txt",
	shell:
		"python {processpath}/find_intersect.py -fcg fred -kg {kgtype}"

rule create_co_ufcg:
	input:
		"{graphpath}/co-occur/ffcg_co/ffcg_co.edgelist",
		"{graphpath}/co-occur/tfcg_co/tfcg_co.edgelist"
	output:
		"{graphpath}/co-occur/ufcg_co/ufcg_co.edgelist",
		"{graphpath}/co-occur/ufcg_co/ufcg_co.graphml"
		"{graphpath}/co-occur/ufcg_co/data/ufcg_co_entities.txt",
		"{graphpath}/co-occur/ufcg_co/data/ufcg_co_node2ID.json",
		"{graphpath}/co-occur/ufcg_co/data/ufcg_co_edgelistID.npy"
	shell:
		"python {createpath}/create_co-occur.py -ft ufcg_co"

rule create_co_ffcg:
	input:
		"{rdfpath}/false_claims.rdf",
	output:
		"{graphpath}/co-occur/ffcg_co/ffcg_co.edgelist",
		"{graphpath}/co-occur/ffcg_co/ffcg_co.graphml"
		"{graphpath}/co-occur/ffcg_co/data/ffcg_co_entities.txt",
		"{graphpath}/co-occur/ffcg_co/data/ffcg_co_node2ID.json",
		"{graphpath}/co-occur/ffcg_co/data/ffcg_co_edgelistID.npy"
	shell:
		"python {createpath}/create_co-occur.py -ft ffcg_co"

rule create_co_tfcg:
	input:
		"{rdfpath}/true_claims.rdf",
	output:
		"{graphpath}/co-occur/tfcg_co/tfcg_co.edgelist",
		"{graphpath}/co-occur/tfcg_co/tfcg_co.graphml"
		"{graphpath}/co-occur/tfcg_co/data/tfcg_co_entities.txt",
		"{graphpath}/co-occur/tfcg_co/data/tfcg_co_node2ID.json",
		"{graphpath}/co-occur/tfcg_co/data/tfcg_co_edgelistID.npy"
	shell:
		"python {createpath}/create_co-occur.py -ft tfcg_co"

rule create_fred_ufcg:
	input:
		"{graphpath}/fred/tfcg/tfcg.edgelist",
		"{graphpath}/fred/ffcg/ffcg.edgelist"
	output:
		"{graphpath}/fred/ufcg/ufcg.edgelist",
		"{graphpath}/fred/ufcg/ufcg.graphml"
		"{graphpath}/fred/ufcg/data/ufcg_entities.txt",
		"{graphpath}/fred/ufcg/data/ufcg_node2ID.json",
		"{graphpath}/fred/ufcg/data/ufcg_edgelistID.npy"
	shell:
		"python {createpath}/create_fred.py -ft ufcg"

rule create_fred_ffcg:
	input:
		"{rdfpath}/false_claims.csv",
	output:
		"{graphpath}/fred/ffcg/ffcg.edgelist",
		"{graphpath}/fred/ffcg/ffcg.graphml"
		"{graphpath}/fred/ffcg/data/ffcg_entities.txt",
		"{graphpath}/fred/ffcg/data/ffcg_node2ID.json",
		"{graphpath}/fred/ffcg/data/ffcg_edgelistID.npy"
	shell:
		"python {createpath}/create_fred.py -ft ffcg"

rule create_fred_tfcg:
	input:
		"{rdfpath}/true_claims.csv",
	output:
		"{graphpath}/fred/tfcg/tfcg.edgelist",
		"{graphpath}/fred/tfcg/tfcg.graphml"
		"{graphpath}/fred/tfcg/data/tfcg_entities.txt",
		"{graphpath}/fred/tfcg/data/tfcg_node2ID.json",
		"{graphpath}/fred/tfcg/data/tfcg_edgelistID.npy"
	shell:
		"python {createpath}/create_fred.py -ft tfcg"

rule create_dbpedia:
	input:
		"{graphpath}/kg/dbpedia/raw/dbpedia_2016-10.nt",
		"{graphpath}/kg/dbpedia/raw/instance_types_en.ttl",
		"{graphpath}/kg/dbpedia/raw/mappingbased_objects_en.ttl",
	output:
		"{graphpath}/kg/dbpedia/dbpedia.nt",
		"{graphpath}/kg/dbpedia/dbpedia.edgelist",
		"{graphpath}/kg/dbpedia/data/dbpedia_nodes.txt",
		"{graphpath}/kg/dbpedia/data/dbpedia_entities.txt",
		"{graphpath}/kg/dbpedia/data/dbpedia_node2ID.json",
		"{graphpath}/kg/dbpedia/data/dbpedia_edgelistID.npy"
	shell:
		"python {createpath}/create_dbpedia.py"
