package targetedbeast.tools;

import java.io.File;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import beast.base.core.BEASTInterface;
import beast.pkgmgmt.BEASTClassLoader;
import targetedbeast.edgeweights.LearnedWeights;

final class BeastCliSupport {

    private BeastCliSupport() {
    }

    static void registerLocalPackageServices(Class<?> anchorClass, File xmlFile) {
        File versionFile = findVersionFile(anchorClass, xmlFile);
        if (versionFile != null) {
            BEASTClassLoader.addServices(versionFile.getAbsolutePath());
        }
    }

    static LearnedWeights findLearnedWeights(BEASTInterface root, String preferredId) {
        List<LearnedWeights> found = findAllObjects(root, LearnedWeights.class);

        if (found.isEmpty()) {
            return null;
        }
        if (preferredId != null && !preferredId.isEmpty()) {
            for (LearnedWeights learnedWeights : found) {
                if (preferredId.equals(learnedWeights.getID())) {
                    return learnedWeights;
                }
            }
        }
        return found.get(0);
    }

    static <T> T findObject(BEASTInterface root, Class<T> type, String preferredId) {
        List<T> found = findAllObjects(root, type);
        if (found.isEmpty()) {
            return null;
        }

        if (preferredId != null && !preferredId.isEmpty()) {
            for (T candidate : found) {
                if (candidate instanceof BEASTInterface beastObject && preferredId.equals(beastObject.getID())) {
                    return candidate;
                }
            }
            return null;
        }

        return found.get(0);
    }

    private static <T> List<T> findAllObjects(BEASTInterface root, Class<T> type) {
        List<T> found = new ArrayList<>();
        Set<BEASTInterface> seen = new HashSet<>();
        Deque<BEASTInterface> queue = new ArrayDeque<>();
        queue.add(root);

        while (!queue.isEmpty()) {
            BEASTInterface current = queue.removeFirst();
            if (!seen.add(current)) {
                continue;
            }

            if (type.isInstance(current)) {
                found.add(type.cast(current));
            }

            for (BEASTInterface child : current.listActiveBEASTObjects()) {
                if (child != null) {
                    queue.addLast(child);
                }
            }
        }
        return found;
    }

    private static File findVersionFile(Class<?> anchorClass, File xmlFile) {
        for (File candidate : candidateVersionFiles(anchorClass, xmlFile)) {
            if (candidate != null && candidate.exists() && candidate.isFile()) {
                return candidate;
            }
        }
        return null;
    }

    private static List<File> candidateVersionFiles(Class<?> anchorClass, File xmlFile) {
        List<File> candidates = new ArrayList<>();

        if (xmlFile != null) {
            File xmlParent = xmlFile.getAbsoluteFile().getParentFile();
            while (xmlParent != null) {
                candidates.add(new File(xmlParent, "targetedbeast/version.xml"));
                candidates.add(new File(xmlParent, "version.xml"));
                xmlParent = xmlParent.getParentFile();
            }
        }

        try {
            Path codeSource = Paths.get(anchorClass.getProtectionDomain().getCodeSource().getLocation().toURI());
            Path current = codeSource.toFile().isDirectory() ? codeSource : codeSource.getParent();
            while (current != null) {
                candidates.add(current.resolve("version.xml").toFile());
                current = current.getParent();
            }
        } catch (URISyntaxException | IllegalStateException ignored) {
            // Fall back to cwd-based probing below.
        }

        Path cwd = Paths.get("").toAbsolutePath();
        while (cwd != null) {
            candidates.add(cwd.resolve("version.xml").toFile());
            cwd = cwd.getParent();
        }
        return candidates;
    }
}