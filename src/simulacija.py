import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import copy
from matplotlib.patches import FancyArrowPatch
import time
from networkx.algorithms import community

#Faza 1: Generisanje mreže
def generisi_mrezu(N=300, m=3):

    G_undirected = nx.barabasi_albert_graph(n=N, m=m, seed=42)
    
    G = nx.DiGraph()
    G.add_nodes_from(G_undirected.nodes())
    
    for n in G.nodes():
        G.nodes[n]['type'] = 'E' if np.random.rand() < 0.8 else 'I'
    
    for u, v in G_undirected.edges():
        if np.random.rand() < 0.5:
            src, dst = u, v
        else:
            src, dst = v, u
            

        tip_izvora = G.nodes[src]['type']
        if tip_izvora == 'E':
           
            tezina = np.random.uniform(0.1, 1.0)
        else:
          
            tezina = np.random.uniform(-1.0, -0.1)
            
        G.add_edge(src, dst, weight=tezina)
        
    return G 

N = 300
G = generisi_mrezu(N)
print(f"Generisana mreza sa {G.number_of_nodes()} cvorova i {G.number_of_edges()} veza.")

degrees_dict = dict(G.degree())
degrees_list = [d for n, d in G.degree()]

sorted_degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
top_hub_node = sorted_degrees[0][0] 
periferni_node = sorted_degrees[-1][0] 

print(f"Najveci HUB je cvor {top_hub_node} sa stepenom {sorted_degrees[0][1]}.")
print(f"Najmanji periferni cvor je {periferni_node} sa stepenom {sorted_degrees[-1][1]}.")

#centralnost
bc = nx.betweenness_centrality(G)
top_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)[0]
print(f"Cvor sa najvecom Betweenness centralnoscu je {top_bc[0]} (vrijednost: {top_bc[1]:.4f})")
#klasterisanje
avg_clustering = nx.average_clustering(G.to_undirected())
print(f"Prosjecni koeficijent klasterisanja: {avg_clustering:.4f}")
#modularnost
communities = community.greedy_modularity_communities(G.to_undirected())
print(f"Broj detektovanih funkcionalnih modula (zajednica): {len(communities)}")

plt.figure(figsize=(10, 5))
plt.hist(degrees_list, bins=30, color='skyblue', edgecolor='black')
plt.title("Distribucija stepena čvorova (Barabási-Albert model)")
plt.xlabel("Stepen čvora (k)")
plt.ylabel("Broj čvorova")
plt.yscale('log') 
plt.grid(True, alpha=0.3)
plt.show()

# Kategorizacija za statički prikaz
num_hubs = int(N * 0.05)
num_mid = int(N * 0.15)
hubs = [n for n, d in sorted_degrees[:num_hubs]]
mid_nodes = [n for n, d in sorted_degrees[num_hubs:num_hubs+num_mid]]

node_colors_static = []
for n in G.nodes():
    if n in hubs: node_colors_static.append('#FF3333')
    elif n in mid_nodes: node_colors_static.append('#FFCC33')
    else: node_colors_static.append('#3399FF')

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, k=0.3, seed=42)

node_sizes = [degrees_dict[n]*15 for n in G.nodes()] 


nx.draw_networkx_edges(
    G, pos, 
    alpha=0.1, 
    edge_color='gray', 
    arrows=True, 
    arrowsize=10, 
    node_size=node_sizes, 
    connectionstyle="arc3,rad=0.05"
)


nx.draw_networkx_nodes(
    G, pos, 
    node_size=node_sizes, 
    node_color=node_colors_static, 
    alpha=0.8
)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Hub neuroni (Gornjih 5%)',
           markerfacecolor='#FF3333', markersize=12),
    Line2D([0], [0], marker='o', color='w', label='Srednji neuroni (5-20%)',
           markerfacecolor='#FFCC33', markersize=9),
    Line2D([0], [0], marker='o', color='w', label='Periferni neuroni (80%)',
           markerfacecolor='#3399FF', markersize=6)
]

plt.legend(
    handles=legend_elements, 
    title="Kategorije čvorova", 
    fontsize=10,
    loc='upper right',           
    bbox_to_anchor=(1.04, 1.06)   
)

plt.title("Topološka struktura mreže")
plt.axis('off')
plt.show()

# Faza 2: Simulacija dinamike širenja
def simuliraj_dinamiku(G, pocetni_cvor, koraci=50, pocetni_theta=0.35, disipacija=0.15, eta=0.08):
    tezin_kopija = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
    energija = {n: 1.0 for n in G.nodes()}
    stanja = {n: 0 for n in G.nodes()}

    stanja[pocetni_cvor] = 1
    istorija = []

    for t in range(koraci):
        nova_stanja = {n: 0 for n in G.nodes()}
        
        aktivne_veze_u_koraku = []

        for i in G.nodes():
            if stanja[i] != 0: continue

            suma_ulaza = sum(tezin_kopija[(j, i)] * stanja[j] for j in G.predecessors(i))
            neto = suma_ulaza - disipacija

            if energija[i] > 0.85 and abs(neto) > pocetni_theta:
                nova_stanja[i] = 1 if neto > 0 else -1
                energija[i] = 0.0

                for j in G.predecessors(i):
                    if stanja[j] != 0:
                        tezin_kopija[(j, i)] += eta
                        tezin_kopija[(j, i)] = np.clip(tezin_kopija[(j, i)], -2.0, 2.0)

                        aktivne_veze_u_koraku.append((j, i))

        istorija.append({'stanja': stanja.copy(), 'putanje': aktivne_veze_u_koraku})
        stanja = nova_stanja.copy()

        if sum(abs(s) for s in stanja.values()) == 0 and t > 5: break

    return istorija

def pokreni_animaciju(G, istorija, naslov="Dinamika"):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(11, 8), facecolor='black')
    ax.set_facecolor('black')
    particles = ax.scatter([], [], s=10, color='white', alpha=0.9, zorder=7)

    velicina_cvorova = 25

    print("Pripremam pozadinu...")
    nx.draw_networkx_edges(
        G, pos_anim,
        alpha=0.03,
        edge_color='gray',
        width=0.5,
        arrows=True,
        arrowsize=7,
        ax=ax,
        node_size=velicina_cvorova,
        connectionstyle="arc3,rad=0.1"
    )

    cvorovi_objekt = nx.draw_networkx_nodes(
        G, pos_anim,
        node_size=velicina_cvorova,
        node_color=['#111111']*len(G.nodes()),
        ax=ax,
        edgecolors='#222222',
        linewidths=0.3
    )
    time.sleep(3)
    interp_steps = 20  
    interval_ms = 30   

    def update(frame):
        step_idx = frame // interp_steps
        intra_step = frame % interp_steps

        if step_idx >= len(istorija):
            particles.set_offsets(np.empty((0, 2)))
            return cvorovi_objekt, particles

        podaci = istorija[step_idx]

        if intra_step == 0:
            nove_boje = ['#111111'] * len(G.nodes())
            for n, s in podaci['stanja'].items():
                if s == 1: nove_boje[n] = '#FF3333'
                elif s == -1: nove_boje[n] = '#3333FF'
            cvorovi_objekt.set_facecolors(nove_boje)

        aktivni_parovi = podaci['putanje']
        coords = []
        if aktivni_parovi:
            for u, v in aktivni_parovi:
                start = np.array(pos_anim[u])
                end = np.array(pos_anim[v])

                progress = intra_step / interp_steps
                if 0 <= progress <= 1:
                    point = (1 - progress) * start + progress * end
                    coords.append(point)

        if coords:
            particles.set_offsets(coords)
            particles.set_alpha(0.9)
        else:
            particles.set_offsets(np.empty((0, 2)))

        return cvorovi_objekt, particles

    ani = FuncAnimation(
        fig, update,
        frames=len(istorija) * interp_steps,
        interval=interval_ms,
        blit=True,
        repeat=False
    )

    plt.title(naslov, color='white', pad=20)
    plt.axis('off')
    plt.show()

pos_anim = nx.spring_layout(G, k=0.4, seed=42)

def sacuvaj_snapshots(G, istorija, pos, prefix="hub_snapshot", koraci_snapshot=[0,5,10,15]):
    degrees_dict = dict(G.degree())
    velicine_dict = {n: 40 + degrees_dict[n]*8 for n in G.nodes()}  

    for t in koraci_snapshot:
        if t >= len(istorija):
            continue
        
        podaci = istorija[t]
        
        boje = []
        for n in G.nodes():
            s = podaci['stanja'][n]
            if s == 1:
                boje.append('#FF3333') 
            elif s == -1:
                boje.append('#3333FF')  
            else:
                boje.append('#AAAAAA')  
        
        plt.figure(figsize=(12,10))
        ax = plt.gca()
        
        nx.draw_networkx_edges(G, pos, alpha=0.03, edge_color='gray', arrows=False)
        
        nx.draw_networkx_nodes(G, pos, node_color=boje, node_size=list(velicine_dict.values()),
                               alpha=0.9, edgecolors='#555555', linewidths=0.5)
        
        aktivne_veze = podaci['putanje']
        if aktivne_veze:
            for u, v in aktivne_veze:
                radius_u = np.sqrt(velicine_dict[u]/np.pi) * 0.015
                radius_v = np.sqrt(velicine_dict[v]/np.pi) * 0.015
                
                arrow = FancyArrowPatch(
                    posA=pos[u], posB=pos[v],
                    arrowstyle='-|>',       
                    color='red',
                    mutation_scale=15,
                    lw=1.8,
                    alpha=0.9,
                    connectionstyle="arc3,rad=0.1", 
                    shrinkA=radius_u,
                    shrinkB=radius_v
                )
                ax.add_patch(arrow)
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.axis('off')
        plt.title(f"Vizualizacija širenja: Korak {t}", fontsize=14)
        plt.savefig(f"{prefix}_{t:02d}.png", dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Generisana slika: {prefix}_{t:02d}.png")

# --- analiza otpornosti ili Faza 3 ---
def analiza_otpornosti(G_original, procenat_uklanjanja=0.4, korak=0.02):
    print("\nPokrećem analizu otpornosti (LSCC)...")
    N = G_original.number_of_nodes()
    koraci = np.arange(0, procenat_uklanjanja + korak, korak)
    
    lscc_nasumicno = []
    lscc_ciljano = []
    svi_cvorovi = list(G_original.nodes())
    np.random.shuffle(svi_cvorovi)
    for p in koraci:
        temp_G = copy.deepcopy(G_original)
        temp_G.remove_nodes_from(svi_cvorovi[:int(p * N)])
        komp = sorted(nx.strongly_connected_components(temp_G), key=len, reverse=True)
        lscc_nasumicno.append(len(komp[0]) / N if komp else 0)

    bc_scores = nx.betweenness_centrality(G_original)
    poredani_cvorovi = [n for n, v in sorted(bc_scores.items(), key=lambda x: x[1], reverse=True)]
    for p in koraci:
        temp_G = copy.deepcopy(G_original)
        temp_G.remove_nodes_from(poredani_cvorovi[:int(p * N)])
        komp = sorted(nx.strongly_connected_components(temp_G), key=len, reverse=True)
        lscc_ciljano.append(len(komp[0]) / N if komp else 0)

    plt.figure(figsize=(10, 6))
    plt.plot(koraci * 100, lscc_nasumicno, 'o-', label='Nasumično uklanjanje (Senzorna smrt)', color='green')
    plt.plot(koraci * 100, lscc_ciljano, 's-', label='Ciljani napad (Oštećenje HUB-ova)', color='red')
    plt.title("Analiza otpornosti neuronske mreže: Održivost komunikacije")
    plt.xlabel("Procenat uklonjenih neurona (%)")
    plt.ylabel("Relativna veličina LSCC (Funkcionalnost)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

print("\nPokrećem Simulaciju A (HUB)... Gledajte veze koje zasvijetle.")
start_time = time.time()
istorija_hub = simuliraj_dinamiku(G, pocetni_cvor=top_hub_node)
end_time = time.time()
print(f"Scenario A (HUB) trajao je {end_time - start_time:.3f} sekundi.")
pos_anim = nx.spring_layout(G, k=0.4, seed=42)

n_snap = 4
koraci_snapshot = np.linspace(0, len(istorija_hub)-1, n_snap, dtype=int)
sacuvaj_snapshots(G, istorija_hub, pos_anim, prefix="hub_snapshot", koraci_snapshot=koraci_snapshot)

pokreni_animaciju(G, istorija_hub, naslov="Scenario A: Aktivacija iz HUB-a")

start_time = time.time()
istorija_periferija = simuliraj_dinamiku(G, pocetni_cvor=periferni_node)
end_time = time.time()
print(f"Scenario B (Periferija) trajao je {end_time - start_time:.3f} sekundi.")
pos_anim = nx.spring_layout(G, k=0.4, seed=42)


n_snap = 4
koraci_snapshot = np.linspace(0, len(istorija_periferija)-1, n_snap, dtype=int)

istorija_periferija = simuliraj_dinamiku(G, pocetni_cvor=periferni_node)

sacuvaj_snapshots(G, istorija_periferija, pos_anim, prefix="periferija_snapshot", koraci_snapshot=koraci_snapshot)

pokreni_animaciju(
    G, 
    istorija_periferija, 
    naslov="Scenario B: Aktivacija iz periferije"
)

analiza_otpornosti(G)

#analiza putanja
def analiza_putanja(G_original, procenat_uklanjanja=0.2, korak=0.02):
    N = G_original.number_of_nodes()
    koraci = np.arange(0, procenat_uklanjanja + korak, korak)
    
    avg_put_ciljano = []
    
    bc_scores = nx.betweenness_centrality(G_original)
    poredani_cvorovi = [n for n, v in sorted(bc_scores.items(), key=lambda x: x[1], reverse=True)]

    for p in koraci:
        temp_G = copy.deepcopy(G_original)
        temp_G.remove_nodes_from(poredani_cvorovi[:int(p * N)])
        
        komp = sorted(nx.strongly_connected_components(temp_G), key=len, reverse=True)
        if komp and len(komp[0]) > 1:
            lscc_subgraph = temp_G.subgraph(komp[0])
            avg_l = nx.average_shortest_path_length(lscc_subgraph)
            avg_put_ciljano.append(avg_l)
        else:
            avg_put_ciljano.append(None) 

    return koraci * 100, avg_put_ciljano

procenti, putanje = analiza_putanja(G)
plt.plot(procenti, putanje, 's-', color='red')
plt.title("Degradacija efikasnosti komunikacije (Ciljani napad)")
plt.xlabel("Procenat uklonjenih hub-ova (%)")
plt.ylabel("Prosječna dužina najkraćeg puta")
plt.show()