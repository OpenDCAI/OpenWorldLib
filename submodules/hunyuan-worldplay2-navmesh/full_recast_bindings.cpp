// full_recast_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <random>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <cstring>

#include "navmesh_builder.h"

#include "DetourNavMesh.h"
#include "DetourNavMeshQuery.h"
#include "DetourCommon.h"
#include "DetourStatus.h"

namespace py = pybind11;

// Wrapper class exposing navmesh build and path query
class RecastNavMeshWrapper {
public:
    RecastNavMeshWrapper()
        : m_nav(nullptr),
          m_query(nullptr)
    {
        m_bmin[0]=m_bmin[1]=m_bmin[2]=0.0f;
        m_bmax[0]=m_bmax[1]=m_bmax[2]=0.0f;
        m_queryExtents[0] = 2.0f;
        m_queryExtents[1] = 4.0f;
        m_queryExtents[2] = 2.0f;
    }

    ~RecastNavMeshWrapper() {
        reset();
    }

    void reset() {
        if (m_query) m_query.reset();
        if (m_nav) { freeDetourMesh(m_nav); m_nav = nullptr; }
        m_polymesh_verts.clear();
        m_polymesh_tris.clear();
    }

    int add(int a, int b) { return a + b; }

    // build navmesh and also capture polymesh (verts/tris) exported by navmesh_builder
    bool build_from_vertices(const std::vector<float>& vertices,
                             const std::vector<int>& indices,
                             float cellSize = 0.03f,
                             float cellHeight = 0.02f,
                             float agentHeight = 2.0f,
                             float agentRadius = 0.6f,
                             float agentMaxClimb = 0.9f,
                             float maxSlope = 45.0f)
    {
        // release existing
        reset();

        // compute and store bounds from input vertices so we can sample random points later
        if (!vertices.empty()) {
            const int nverts = static_cast<int>(vertices.size() / 3);
            m_bmin[0] = m_bmin[1] = m_bmin[2] =  std::numeric_limits<float>::infinity();
            m_bmax[0] = m_bmax[1] = m_bmax[2] = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < nverts; ++i) {
                const float* v = &vertices[i*3];
                for (int k = 0; k < 3; ++k) {
                    if (v[k] < m_bmin[k]) m_bmin[k] = v[k];
                    if (v[k] > m_bmax[k]) m_bmax[k] = v[k];
                }
            }
            const float pad = 0.01f;
            m_bmin[0] -= pad; m_bmin[1] -= pad; m_bmin[2] -= pad;
            m_bmax[0] += pad; m_bmax[1] += pad; m_bmax[2] += pad;
        } else {
            m_bmin[0]=m_bmin[1]=m_bmin[2]=0.0f;
            m_bmax[0]=m_bmax[1]=m_bmax[2]=0.0f;
        }

        // call builder and request exported polymesh
        std::vector<float> outVerts;
        std::vector<int> outTris;
        dtNavMesh* nav = buildDetourMeshFromTriangles(vertices, indices,
                                                      cellSize, cellHeight,
                                                      agentHeight, agentRadius,
                                                      agentMaxClimb, maxSlope,
                                                      &outVerts, &outTris);
        if (!nav) {
            throw std::runtime_error("build_from_vertices: failed to build dtNavMesh");
        }

        m_nav = nav;
        // store exported polymesh locally for get_polymesh
        m_polymesh_verts = std::move(outVerts);
        m_polymesh_tris = std::move(outTris);

        // allocate and init query
        m_query = std::unique_ptr<dtNavMeshQuery>(new dtNavMeshQuery());
        if (!m_query) {
            freeDetourMesh(m_nav);
            m_nav = nullptr;
            throw std::runtime_error("build_from_vertices: failed to allocate dtNavMeshQuery");
        }
        dtStatus status = m_query->init(m_nav, 2048);
        if (dtStatusFailed(status)) {
            m_query.reset();
            freeDetourMesh(m_nav);
            m_nav = nullptr;
            throw std::runtime_error("build_from_vertices: failed to init dtNavMeshQuery");
        }

        return true;
    }

    // set query extents used by findNearestPoly etc.
    void set_query_extents(float ex, float ey, float ez) {
        m_queryExtents[0] = ex;
        m_queryExtents[1] = ey;
        m_queryExtents[2] = ez;
    }

    std::vector<float> find_path(float sx, float sy, float sz,
                                 float ex, float ey, float ez,
                                 int maxPolys = 512, int maxSmooth = 512)
    {
        std::vector<float> out;
        if (!m_nav || !m_query) {
            throw std::runtime_error("find_path: nav or query is null");
        }

        dtQueryFilter filter;
        filter.setIncludeFlags(0xffff);
        filter.setExcludeFlags(0);

        const float startPos[3] = { sx, sy, sz };
        const float endPos[3]   = { ex, ey, ez };

        // find nearest polys
        dtPolyRef startRef = 0, endRef = 0;
        float startNearest[3], endNearest[3];
        dtStatus s1 = m_query->findNearestPoly(startPos, m_queryExtents, &filter, &startRef, startNearest);
        dtStatus s2 = m_query->findNearestPoly(endPos, m_queryExtents, &filter, &endRef, endNearest);
        if (dtStatusFailed(s1) || dtStatusFailed(s2) || !startRef || !endRef) {
            return out;
        }

        // get path polygon refs
        std::vector<dtPolyRef> polys(maxPolys);
        int npolys = 0;
        dtStatus spath = m_query->findPath(startRef, endRef, startNearest, endNearest, &filter, polys.data(), &npolys, maxPolys);
        if (dtStatusFailed(spath) || npolys == 0) {
            return out;
        }

        // straight path
        std::vector<float> straight(maxSmooth * 3);
        std::vector<unsigned char> straightFlags(maxSmooth);
        std::vector<dtPolyRef> straightPolys(maxSmooth);
        int nstraight = 0;
        dtStatus sstraight = m_query->findStraightPath(startNearest, endNearest, polys.data(), npolys,
                                                       straight.data(), straightFlags.data(), straightPolys.data(), &nstraight, maxSmooth);
        if (dtStatusFailed(sstraight) || nstraight <= 0) {
            return out;
        }

        out.reserve(nstraight * 3);
        for (int i = 0; i < nstraight; ++i) {
            out.push_back(straight[i*3 + 0]);
            out.push_back(straight[i*3 + 1]);
            out.push_back(straight[i*3 + 2]);
        }
        return out;
    }

    // return a random point projected to navmesh (or empty vector if not found)
    std::vector<float> get_random_point(int max_tries = 64) {
        std::vector<float> p;
        if (!m_nav || !m_query) {
            throw std::runtime_error("get_random_point: nav or query is null");
        }

        if (m_bmin[0] >= m_bmax[0] || m_bmin[1] >= m_bmax[1] || m_bmin[2] >= m_bmax[2]) {
            return p;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distX(m_bmin[0], m_bmax[0]);
        std::uniform_real_distribution<float> distY(m_bmin[1], m_bmax[1]);
        std::uniform_real_distribution<float> distZ(m_bmin[2], m_bmax[2]);

        dtQueryFilter filter;
        filter.setIncludeFlags(0xffff);
        filter.setExcludeFlags(0);

        for (int t = 0; t < max_tries; ++t) {
            float sample[3];
            sample[0] = distX(gen);
            sample[1] = distY(gen);
            sample[2] = distZ(gen);

            dtPolyRef polyRef = 0;
            float nearest[3] = {0.0f, 0.0f, 0.0f};
            dtStatus st = m_query->findNearestPoly(sample, m_queryExtents, &filter, &polyRef, nearest);
            if (!dtStatusFailed(st) && polyRef) {
                p.push_back(nearest[0]);
                p.push_back(nearest[1]);
                p.push_back(nearest[2]);
                return p;
            }
        }
        return p;
    }

    std::string get_version() {
        return "RecastNavigation Python Bindings (custom)";
    }

    // return stored polymesh as numpy arrays (verts float32 flat, tris int32 flat)
    py::tuple get_polymesh() {
        // build numpy arrays from stored vectors
        py::array_t<float> verts_np( m_polymesh_verts.size() );
        if (!m_polymesh_verts.empty()) {
            std::memcpy(verts_np.mutable_data(), m_polymesh_verts.data(), m_polymesh_verts.size() * sizeof(float));
        }
        py::array_t<int> tris_np( m_polymesh_tris.size() );
        if (!m_polymesh_tris.empty()) {
            std::memcpy(tris_np.mutable_data(), m_polymesh_tris.data(), m_polymesh_tris.size() * sizeof(int));
        }
        return py::make_tuple(verts_np, tris_np);
    }

    // project arbitrary point to navmesh: returns (x,y,z, polyRef) or None
    py::object project_point(float x, float y, float z) {
        if (!m_nav || !m_query) return py::none();

        dtQueryFilter filter;
        filter.setIncludeFlags(0xffff);
        filter.setExcludeFlags(0);

        float pos[3] = { x, y, z };
        dtPolyRef poly = 0;
        float nearest[3] = {0.0f, 0.0f, 0.0f};

        dtStatus st = m_query->findNearestPoly(pos, m_queryExtents, &filter, &poly, nearest);
        if (dtStatusFailed(st) || !poly) {
            return py::none();
        }

        bool posOverPoly = false;
        dtStatus cpst = m_query->closestPointOnPoly(poly, pos, nearest, &posOverPoly);
        if (dtStatusFailed(cpst)) {
            return py::none();
        }

        unsigned long long polyid = (unsigned long long)poly;
        return py::make_tuple(nearest[0], nearest[1], nearest[2], polyid);
    }

private:
    dtNavMesh* m_nav;
    std::unique_ptr<dtNavMeshQuery> m_query;
    float m_bmin[3];
    float m_bmax[3];
    float m_queryExtents[3];

    // exported polymesh captured at build time
    std::vector<float> m_polymesh_verts; // flat x,y,z,...
    std::vector<int>   m_polymesh_tris;  // flat triplets of indices
};

PYBIND11_MODULE(recast, m) {
    py::class_<RecastNavMeshWrapper>(m, "RecastNavMesh")
        .def(py::init<>())
        .def("add", &RecastNavMeshWrapper::add)
        .def("build_from_vertices",
             &RecastNavMeshWrapper::build_from_vertices,
             py::arg("vertices"), py::arg("indices"),
             py::arg("cellSize") = 0.03f, py::arg("cellHeight") = 0.02f,
             py::arg("agentHeight") = 2.0f, py::arg("agentRadius") = 0.6f,
             py::arg("agentMaxClimb") = 0.9f, py::arg("maxSlope") = 45.0f)
        .def("find_path", &RecastNavMeshWrapper::find_path,
             py::arg("sx"), py::arg("sy"), py::arg("sz"),
             py::arg("ex"), py::arg("ey"), py::arg("ez"),
             py::arg("maxPolys") = 512, py::arg("maxSmooth") = 512)
        .def("get_random_point", &RecastNavMeshWrapper::get_random_point,
             py::arg("max_tries") = 64)
        .def("get_polymesh", &RecastNavMeshWrapper::get_polymesh)
        .def("project_point", &RecastNavMeshWrapper::project_point)
        .def("get_version", &RecastNavMeshWrapper::get_version)
        .def("reset", &RecastNavMeshWrapper::reset)
        .def("set_query_extents", &RecastNavMeshWrapper::set_query_extents,
             py::arg("ex"), py::arg("ey"), py::arg("ez"));
}