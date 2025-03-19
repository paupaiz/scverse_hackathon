"""
Calculate a ligand-receptor co-expression score between neighboring cells.
"""

import itertools
from typing import Dict, List, Set, Tuple, Union, Optional
from omnipath.interactions import import_intercell_network
import numpy as np
from anndata import AnnData


import itertools
from typing import Dict, Optional, Set, Tuple
from tqdm import tqdm
import numpy as np
from anndata import AnnData

def ligand_receptor_score(
    adata: AnnData,
    cell_to_neighbors: Dict[str, Set[str]],
    lr_pairs: Union[Tuple[str, str], List[Tuple[str, str]]]
) -> Dict[Tuple[str, str], float]:
    """
    Calculates an average co-expression score of a ligand-receptor pair between neighboring cells.
    
    For each ligand–receptor pair, the score is computed as the average of the square-root of 
    the product of the expression values of the ligand (in the first cell) and receptor (in the second cell)
    across all neighboring cell pairs. If no neighbor pairs are found, np.nan is returned for that pair.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    cell_to_neighbors : dict
        A dictionary mapping each cell to a set of its neighboring cells.
    lr_pairs : tuple or list of tuples
        One or more ligand–receptor pairs specified as (ligand, receptor). If a single tuple is provided,
        it is converted into a list.
        
    Returns
    -------
    dict
        A dictionary mapping each ligand–receptor pair to its co-expression score.
    """
    # Generate neighbor pairs as a NumPy array of (cell, neighbor) tuples
    neighbor_pairs = np.array(
        list(
            itertools.chain.from_iterable(
                [
                    [(cell, neighbor) for neighbor in neighbors]
                    for cell, neighbors in cell_to_neighbors.items()
                ]
            )
        )
    )

    def calculate_score(ligand: str, receptor: str) -> float:
        # If no neighbor pairs exist, return np.nan
        if neighbor_pairs.size == 0:
            return np.nan
        # Otherwise, extract expression values from the AnnData object.
        # Here, neighbor_pairs.T[0] are the ligand cells and neighbor_pairs.T[1] the receptor cells.
        ligand_values = adata[neighbor_pairs.T[0], ligand].X.toarray()
        receptor_values = adata[neighbor_pairs.T[1], receptor].X.toarray()
        score = np.sum(np.sqrt(ligand_values * receptor_values)) / neighbor_pairs.shape[0]
        return score

    # Ensure lr_pairs is a list even if a single tuple is provided.
    if isinstance(lr_pairs, tuple):
        lr_pairs = [lr_pairs]

    # Compute the score for each ligand–receptor pair.
    lr_pair_to_score = {(ligand, receptor): calculate_score(ligand, receptor)
                        for ligand, receptor in lr_pairs}
    return lr_pair_to_score

def ligand_receptor_score_per_niche(
    adata: AnnData,
    niche_to_cell_to_neighbors: Dict[str, Set[str]],
    lr_pairs: Optional[Union[Tuple[str, str], List[Tuple[str, str]]]] = None,
) -> Dict[str, Dict[Tuple[str, str], float]]:
    """
    Calculates an average co-expression score of a
    ligand-receptor pair between neighboring cells within each cellular niche
    calculated by :func:`monkeybread.calc.cellular_niches`. Statistical test is as described 
    in :cite:p:`He2021.11.03.467020` (See Figure 4). This function is a wrapper around 
    :func:`monkeybread.calc.ligand_receptor_score` and calls this function separately for 
    each niche.

    Parameters
    ----------
    adata
        Annotated data matrix.
    cell_to_neighbors
        A mapping of cells to their neighbors, as calculated by
        :func:`monkeybread.calc.cell_neighbors`.
    lr_pairs
        One or multiple tuples corresponding to (ligand, receptor). If `None` then
        ligand/receptor pairs will be downloaded from Omnipath via the
        :func:`omnipath.interactions.import_intercell_network`.

    Returns
    -------
    A dictionary mapping each niche to a sub-dictionary mapping each ligand-receptor 
    pair to its co-expression score.
    """

    niche_to_lr_pair_to_score = {}
    for niche, cell_to_neighbors in niche_to_cell_to_neighbors.items():
        
        lr_pair_to_score = ligand_receptor_score(
            adata,
            cell_to_neighbors,
            lr_pairs,
        )
        niche_to_lr_pair_to_score[niche] = lr_pair_to_score
    return niche_to_lr_pair_to_score