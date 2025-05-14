import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom
from scipy.spatial.distance import pdist, squareform
import random

class DimensionalityReduction:
    def __init__(self, n_components=2, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.np_rng = np.random.default_rng(random_state)
        random.seed(random_state)

    def preprocess_data(self, X):
        return self.scaler.fit_transform(X)

    def pca(self, X):
        X_scaled = self.preprocess_data(X)
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        return pca.fit_transform(X_scaled)

    def lda(self, X, y):
        X_scaled = self.preprocess_data(X)
        n_classes = np.unique(y).shape[0]
        max_components = min(X_scaled.shape[1], n_classes - 1)
        if self.n_components > max_components:
            raise ValueError(f"LDA: n_components must be <= {max_components} (min(n_classes-1, n_features))")
        lda = LinearDiscriminantAnalysis(n_components=self.n_components)
        return lda.fit_transform(X_scaled, y)

    def tsne(self, X, perplexity=30):
        X_scaled = self.preprocess_data(X)
        tsne = TSNE(n_components=self.n_components, 
                    perplexity=perplexity,
                    random_state=self.random_state)
        return tsne.fit_transform(X_scaled)

    def mds(self, X):
        X_scaled = self.preprocess_data(X)
        mds = MDS(n_components=self.n_components, random_state=self.random_state)
        return mds.fit_transform(X_scaled)

    def isomap(self, X, n_neighbors=5):
        X_scaled = self.preprocess_data(X)
        isomap = Isomap(n_components=self.n_components, n_neighbors=n_neighbors)
        return isomap.fit_transform(X_scaled)

    def ica(self, X):
        X_scaled = self.preprocess_data(X)
        ica = FastICA(n_components=self.n_components, random_state=self.random_state, max_iter=1000)
        try:
            return ica.fit_transform(X_scaled)
        except Exception as e:
            print(f"[ICA] Warning: {e}")
            return np.zeros((X_scaled.shape[0], self.n_components))  # fallback

    def som(self, X, grid_size=(20, 20)):
        X_scaled = self.preprocess_data(X)
        som = MiniSom(grid_size[0], grid_size[1], X_scaled.shape[1],
                      sigma=1.0, learning_rate=0.5,
                      random_seed=self.random_state)
        som.train_random(X_scaled, 1000)
        coordinates = np.array([som.winner(x) for x in X_scaled])
        return coordinates
    

    def ga_pca(self, X, y, population_size=50, generations=100, mutation_rate=0.1):
        X_scaled = self.preprocess_data(X)
        n_features = X_scaled.shape[1]
        rng = self.np_rng

        def create_individual():
            # Ensure at least one True
            while True:
                ind = rng.random(n_features) > 0.5
                if ind.sum() > 0:
                    return ind

        def fitness(individual):
            if individual.sum() == 0: return 0
            selected_features = X_scaled[:, individual]
            pca = PCA(n_components=min(self.n_components, selected_features.shape[1]))
            reduced = pca.fit_transform(selected_features)
            knn = KNeighborsClassifier(n_neighbors=5)
            scores = []
            for _ in range(5):
                X_train, X_test, y_train, y_test = train_test_split(reduced, y, test_size=0.2, random_state=self.random_state)
                knn.fit(X_train, y_train)
                scores.append(knn.score(X_test, y_test))
            return np.mean(scores)

        def crossover(parent1, parent2):
            pt = rng.integers(1, n_features)
            child1 = np.hstack((parent1[:pt], parent2[pt:]))
            child2 = np.hstack((parent2[:pt], parent1[pt:]))
            return child1, child2

        def mutate(individual):
            ind = individual.copy()
            for i in range(len(ind)):
                if rng.random() < mutation_rate:
                    ind[i] = not ind[i]
            # Ensure at least one feature
            if ind.sum() == 0:
                ind[rng.integers(0, n_features)] = True
            return ind

        population = [create_individual() for _ in range(population_size)]

        for _ in range(generations):
            fitness_scores = [fitness(ind) for ind in population]
            # Tournament selection
            new_population = []
            for _ in range(population_size):
                tournament = rng.choice(population_size, 3)
                winner = tournament[np.argmax([fitness_scores[i] for i in tournament])]
                new_population.append(population[winner].copy())
            # Crossover and mutation
            i = 0
            while i < population_size-1:
                if rng.random() < 0.8:
                    c1, c2 = crossover(new_population[i], new_population[i+1])
                    new_population[i], new_population[i+1] = mutate(c1), mutate(c2)
                else:
                    new_population[i], new_population[i+1] = mutate(new_population[i]), mutate(new_population[i+1])
                i += 2
            if population_size % 2 == 1:
                new_population[-1] = mutate(new_population[-1])
            population = new_population

        # Final selection
        best_individual = population[np.argmax(fitness_scores)]

        # best_individual = population[np.argmax([fitness(ind) for ind in population])] fixed this #1
        selected_features = X_scaled[:, best_individual]
        pca = PCA(n_components=min(self.n_components, selected_features.shape[1]))
        return pca.fit_transform(selected_features)

    def abc_projection(self, X, y, n_bees=50, n_iterations=100, limit=10):
        X_scaled = self.preprocess_data(X)
        n_features = X_scaled.shape[1]
        rng = self.np_rng

        def create_solution():
            return rng.normal(0, 1, (n_features, self.n_components))

        def fitness(solution):
            projected = X_scaled @ solution
            knn = KNeighborsClassifier(n_neighbors=5)
            scores = []
            for _ in range(5):
                X_train, X_test, y_train, y_test = train_test_split(projected, y, test_size=0.2, random_state=self.random_state)
                knn.fit(X_train, y_train)
                scores.append(knn.score(X_test, y_test))
            return np.mean(scores)

        solutions = [create_solution() for _ in range(n_bees)]
        fitnesses = np.array([fitness(sol) for sol in solutions])
        trials = np.zeros(n_bees)

        for _ in range(n_iterations):
            # Employed bees
            for i in range(n_bees):
                k = rng.integers(n_bees)
                while k == i:
                    k = rng.integers(n_bees)
                phi = rng.uniform(-1, 1, solutions[i].shape)
                new_solution = solutions[i] + phi * (solutions[i] - solutions[k])
                new_fitness = fitness(new_solution)
                if new_fitness > fitnesses[i]:
                    solutions[i] = new_solution
                    fitnesses[i] = new_fitness
                    trials[i] = 0
                else:
                    trials[i] += 1
            # Onlookers
            sum_fitness = fitnesses.sum()
            probabilities = fitnesses / sum_fitness if sum_fitness > 0 else np.ones(n_bees) / n_bees
            for i in range(n_bees):
                if rng.random() < probabilities[i]:
                    k = rng.integers(n_bees)
                    while k == i:
                        k = rng.integers(n_bees)
                    phi = rng.uniform(-1, 1, solutions[i].shape)
                    new_solution = solutions[i] + phi * (solutions[i] - solutions[k])
                    new_fitness = fitness(new_solution)
                    if new_fitness > fitnesses[i]:
                        solutions[i] = new_solution
                        fitnesses[i] = new_fitness
                        trials[i] = 0
                    else:
                        trials[i] += 1
            # Scouts
            for i in range(n_bees):
                if trials[i] >= limit:
                    solutions[i] = create_solution()
                    fitnesses[i] = fitness(solutions[i])
                    trials[i] = 0

        best_solution = solutions[np.argmax(fitnesses)]
        return X_scaled @ best_solution

    def aco_feature_selection(self, X, y, n_ants=50, n_iterations=100, alpha=1, beta=2, rho=0.1):
        X_scaled = self.preprocess_data(X)
        n_features = X_scaled.shape[1]
        rng = self.np_rng

        def create_path():
            while True:
                path = rng.random(n_features) > 0.5
                if path.sum() > 0:
                    return path

        def fitness(path):
            if path.sum() == 0: return 0
            selected_features = X_scaled[:, path]
            pca = PCA(n_components=min(self.n_components, selected_features.shape[1]))
            reduced = pca.fit_transform(selected_features)
            knn = KNeighborsClassifier(n_neighbors=5)
            scores = []
            for _ in range(5):
                X_train, X_test, y_train, y_test = train_test_split(reduced, y, test_size=0.2, random_state=self.random_state)
                knn.fit(X_train, y_train)
                scores.append(knn.score(X_test, y_test))
            return np.mean(scores)

        pheromones = np.ones(n_features)
        best_path, best_fitness = None, -np.inf

        for _ in range(n_iterations):
            paths, fitnesses = [], []
            for _ in range(n_ants):
                path = create_path()
                paths.append(path)
                fitnesses.append(fitness(path))
            pheromones *= (1 - rho)
            for i, path in enumerate(paths):
                pheromones[path] += fitnesses[i]
            current_best_idx = np.argmax(fitnesses)
            if fitnesses[current_best_idx] > best_fitness:
                best_fitness = fitnesses[current_best_idx]
                best_path = paths[current_best_idx]

        selected_features = X_scaled[:, best_path]
        pca = PCA(n_components=min(self.n_components, selected_features.shape[1]))
        return pca.fit_transform(selected_features)

    def pso_projection(self, X, y, n_particles=50, n_iterations=100, w=0.7, c1=1.5, c2=1.5):
        X_scaled = self.preprocess_data(X)
        n_features = X_scaled.shape[1]
        rng = self.np_rng

        def create_particle():
            position = rng.standard_normal((n_features, self.n_components))
            velocity = rng.standard_normal((n_features, self.n_components)) * 0.1
            return position, velocity

        def fitness(position):
            projected = X_scaled @ position
            knn = KNeighborsClassifier(n_neighbors=5)
            scores = []
            for _ in range(5):
                X_train, X_test, y_train, y_test = train_test_split(projected, y, test_size=0.2, random_state=self.random_state)
                knn.fit(X_train, y_train)
                scores.append(knn.score(X_test, y_test))
            return np.mean(scores)

        particles = [create_particle() for _ in range(n_particles)]
        best_positions = [p[0].copy() for p in particles]
        best_fitnesses = [fitness(p[0]) for p in particles]
        global_best_idx = np.argmax(best_fitnesses)
        global_best_position = best_positions[global_best_idx].copy()
        global_best_fitness = best_fitnesses[global_best_idx]

        for _ in range(n_iterations):
            for i, (position, velocity) in enumerate(particles):
                r1, r2 = rng.random(), rng.random()
                velocity = (w * velocity +
                            c1 * r1 * (best_positions[i] - position) +
                            c2 * r2 * (global_best_position - position))
                position += velocity
                current_fitness = fitness(position)
                if current_fitness > best_fitnesses[i]:
                    best_fitnesses[i] = current_fitness
                    best_positions[i] = position.copy()
                    if current_fitness > global_best_fitness:
                        global_best_fitness = current_fitness
                        global_best_position = position.copy()
                particles[i] = (position, velocity)
        return X_scaled @ global_best_position

#################################################################

class EvaluationMetrics:
    def __init__(self, k=5):
        self.k = k

    def trustworthiness(self, X_high, X_low, k=5):
        """
        Implementation following sklearn.manifold.trustworthiness (paraphrased)
        """
        from sklearn.utils import check_random_state
        from scipy.sparse import csr_matrix

        n_samples = X_high.shape[0]
        dist_X = squareform(pdist(X_high))
        np.fill_diagonal(dist_X, np.inf)
        neighbors_X = np.argsort(dist_X, axis=1)[:, :k]

        dist_X_low = squareform(pdist(X_low))
        np.fill_diagonal(dist_X_low, np.inf)
        neighbors_X_low = np.argsort(dist_X_low, axis=1)[:, :k]

        ranks_X = np.argsort(np.argsort(dist_X, axis=1), axis=1)

        t = 0.0
        for i in range(n_samples):
            ux = set(neighbors_X_low[i]) - set(neighbors_X[i])
            t += sum(ranks_X[i, j] - k for j in ux)
        t = 1.0 - (2.0 / (n_samples * k * (2 * n_samples - 3 * k - 1))) * t
        return t

    def continuity(self, X_high, X_low, k=5):
        """
        As above, but swaps the roles.
        """
        n_samples = X_high.shape[0]
        dist_X = squareform(pdist(X_high))
        np.fill_diagonal(dist_X, np.inf)
        neighbors_X = np.argsort(dist_X, axis=1)[:, :k]

        dist_X_low = squareform(pdist(X_low))
        np.fill_diagonal(dist_X_low, np.inf)
        neighbors_X_low = np.argsort(dist_X_low, axis=1)[:, :k]

        ranks_X_low = np.argsort(np.argsort(dist_X_low, axis=1), axis=1)

        c = 0.0
        for i in range(n_samples):
            vy = set(neighbors_X[i]) - set(neighbors_X_low[i])
            c += sum(ranks_X_low[i, j] - k for j in vy)
        c = 1.0 - (2.0 / (n_samples * k * (2 * n_samples - 3 * k - 1))) * c
        return c

    def knn_accuracy(self, X_low, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X_low, y, test_size=test_size, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(X_train, y_train)
        return knn.score(X_test, y_test)

    def silhouette_score(self, X_low, y):
        # y should be cluster labels; using class labels is OK only for supervised clustering metrics.
        return silhouette_score(X_low, y)


    def visualize(self, X_low, y, title="Dimensionality Reduction"):
        plt.figure(figsize=(10, 8))
        if X_low.shape[1] == 1:
            # Only one component: plot as a strip plot for better visualization
            scatter = plt.scatter(X_low[:, 0], np.zeros_like(X_low[:, 0]), c=y, cmap='viridis')
            plt.ylabel('Value (n_components=1)')
            plt.xlabel('Component 1')
        else:
            scatter = plt.scatter(X_low[:, 0], X_low[:, 1], c=y, cmap='viridis')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
        plt.colorbar(scatter)
        plt.title(title)
        plt.show()

# Example Usage (Unchanged)
# dr = DimensionalityReduction(n_components=2)
# X_pca = dr.pca(X)
# metrics = EvaluationMetrics(k=5)
# trust = metrics.trustworthiness(X, X_pca)
# cont = metrics.continuity(X, X_pca)
# knn_acc = metrics.knn_accuracy(X_pca, y)
# sil_score = metrics.silhouette_score(X_pca, y)
# metrics.visualize(X_pca, y, title="PCA Visualization")