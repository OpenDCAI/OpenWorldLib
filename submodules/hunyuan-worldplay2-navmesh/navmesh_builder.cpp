#include "navmesh_builder.h"
#include <iostream>
#include <limits>
#include <cstring>
#include <cmath>
#include <vector>
#include <type_traits>

#include "Recast.h"
#include "DetourNavMesh.h"
#include "DetourNavMeshBuilder.h"
#include "DetourCommon.h"
#include "DetourAlloc.h"
#include "DetourNavMeshQuery.h"
#include "DetourStatus.h"

using namespace std;

#ifndef RC_MAX_VERTS_PER_POLY
#define RC_MAX_VERTS_PER_POLY 6
#endif

static int countMarkedTriangles(const unsigned char* areas, int ntris) {
    int cnt = 0;
    for (int i = 0; i < ntris; ++i) if (areas[i] != 0) ++cnt;
    return cnt;
}

static int getCompactSpanCount(const rcCompactHeightfield* chf) {
    int total = 0;
    if (!chf) return 0;
    for (int i = 0; i < chf->width * chf->height; ++i) {
        rcCompactCell c = chf->cells[i];
        total += c.count;
    }
    return total;
}

dtNavMesh* buildDetourMeshFromTriangles(const std::vector<float>& vertices,
                                        const std::vector<int>& indices,
                                        float cellSize,
                                        float cellHeight,
                                        float agentHeight,
                                        float agentRadius,
                                        float agentMaxClimb,
                                        float maxSlope,
                                        std::vector<float>* outVerts,
                                        std::vector<int>* outTris) {
    if (vertices.empty() || indices.empty()) {
        std::cerr << "buildDetourMeshFromTriangles: empty input" << std::endl;
        return nullptr;
    }
    const int nverts = (int)(vertices.size() / 3);
    const int ntris = (int)(indices.size() / 3);
    if (nverts < 3 || ntris < 1) {
        std::cerr << "buildDetourMeshFromTriangles: not enough geometry" << std::endl;
        return nullptr;
    }

    // Compute bounds
    float bmin[3], bmax[3];
    bmin[0] = bmin[1] = bmin[2] = std::numeric_limits<float>::infinity();
    bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < nverts; ++i) {
        const float* v = &vertices[i*3];
        for (int k = 0; k < 3; ++k) {
            bmin[k] = std::min(bmin[k], v[k]);
            bmax[k] = std::max(bmax[k], v[k]);
        }
    }
    const float pad = 0.01f;
    bmin[0] -= pad; bmin[1] -= pad; bmin[2] -= pad;
    bmax[0] += pad; bmax[1] += pad; bmax[2] += pad;

    rcContext ctx;

    // Config
    rcConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    rcVcopy(cfg.bmin, bmin);
    rcVcopy(cfg.bmax, bmax);
    cfg.cs = cellSize;
    cfg.ch = cellHeight;
    cfg.walkableSlopeAngle = maxSlope;
    // --- 修正开始 ---
    cfg.walkableHeight = (int)ceilf(agentHeight / cfg.ch);

    // 计算爬坡体素数
    int climbVoxels = (int)floorf(agentMaxClimb / cfg.ch);
    // 关键修正：如果允许爬坡且计算结果为0，强制设为1，防止路面断裂
    if (agentMaxClimb > 0.0f && climbVoxels == 0) {
        climbVoxels = 1;
    }
    cfg.walkableClimb = climbVoxels;
    // --- 修正结束 ---

    cfg.walkableRadius = (int)ceilf(agentRadius / cfg.cs);
    cfg.maxEdgeLen = (int)(12.0f / cfg.cs);
    cfg.maxSimplificationError = 1.3f;
    cfg.minRegionArea = (int)rcSqr(2);
    cfg.mergeRegionArea = (int)rcSqr(20);
    cfg.maxVertsPerPoly = RC_MAX_VERTS_PER_POLY;
    cfg.tileSize = 0;
    cfg.detailSampleDist = cfg.cs * 6.0f;
    cfg.detailSampleMaxError = 1.0f;

    int gw = 0, gh = 0;
    rcCalcGridSize(cfg.bmin, cfg.bmax, cfg.cs, &gw, &gh);
    if (gw <= 0 || gh <= 0) {
        std::cerr << "buildDetourMeshFromTriangles: invalid grid size gw=" << gw << " gh=" << gh << std::endl;
        return nullptr;
    }

    rcHeightfield* solid = rcAllocHeightfield();
    if (!solid) {
        std::cerr << "buildDetourMeshFromTriangles: rcAllocHeightfield failed" << std::endl;
        return nullptr;
    }
    if (!rcCreateHeightfield(&ctx, *solid, gw, gh, cfg.bmin, cfg.bmax, cfg.cs, cfg.ch)) {
        std::cerr << "buildDetourMeshFromTriangles: rcCreateHeightfield failed" << std::endl;
        rcFreeHeightField(solid);
        return nullptr;
    }

    unsigned char* triAreas = new unsigned char[ntris];
    memset(triAreas, 0, ntris * sizeof(unsigned char));
    rcMarkWalkableTriangles(&ctx, cfg.walkableSlopeAngle, &vertices[0], nverts, &indices[0], ntris, triAreas);

    int marked = countMarkedTriangles(triAreas, ntris);

    if (!rcRasterizeTriangles(&ctx, &vertices[0], nverts, &indices[0], triAreas, ntris, *solid, cfg.walkableClimb)) {
        std::cerr << "buildDetourMeshFromTriangles: rcRasterizeTriangles failed" << std::endl;
        delete [] triAreas;
        rcFreeHeightField(solid);
        return nullptr;
    }
    delete [] triAreas;

    rcCompactHeightfield* chf = rcAllocCompactHeightfield();
    if (!chf) {
        std::cerr << "buildDetourMeshFromTriangles: rcAllocCompactHeightfield failed" << std::endl;
        rcFreeHeightField(solid);
        return nullptr;
    }
    if (!rcBuildCompactHeightfield(&ctx, cfg.walkableHeight, cfg.walkableClimb, *solid, *chf)) {
        std::cerr << "buildDetourMeshFromTriangles: rcBuildCompactHeightfield failed" << std::endl;
        rcFreeCompactHeightfield(chf);
        rcFreeHeightField(solid);
        return nullptr;
    }

    int erodeRadius = cfg.walkableRadius;
    if (erodeRadius > 1) erodeRadius = 1;
    rcErodeWalkableArea(&ctx, erodeRadius, *chf);

    rcBuildDistanceField(&ctx, *chf);
    rcBuildRegions(&ctx, *chf, 0, cfg.minRegionArea, cfg.mergeRegionArea);

    rcContourSet* cset = rcAllocContourSet();
    if (!cset) {
        std::cerr << "buildDetourMeshFromTriangles: rcAllocContourSet failed" << std::endl;
        rcFreeCompactHeightfield(chf);
        rcFreeHeightField(solid);
        return nullptr;
    }
    if (!rcBuildContours(&ctx, *chf, cfg.maxSimplificationError, cfg.maxEdgeLen, *cset)) {
        std::cerr << "buildDetourMeshFromTriangles: rcBuildContours failed" << std::endl;
        rcFreeContourSet(cset);
        rcFreeCompactHeightfield(chf);
        rcFreeHeightField(solid);
        return nullptr;
    }

    rcPolyMesh* pmesh = rcAllocPolyMesh();
    if (!pmesh) {
        std::cerr << "buildDetourMeshFromTriangles: rcAllocPolyMesh failed" << std::endl;
        rcFreeContourSet(cset);
        rcFreeCompactHeightfield(chf);
        rcFreeHeightField(solid);
        return nullptr;
    }
    if (!rcBuildPolyMesh(&ctx, *cset, cfg.maxVertsPerPoly, *pmesh)) {
        std::cerr << "buildDetourMeshFromTriangles: rcBuildPolyMesh failed" << std::endl;
        rcFreePolyMesh(pmesh);
        rcFreeContourSet(cset);
        rcFreeCompactHeightfield(chf);
        rcFreeHeightField(solid);
        return nullptr;
    }

    rcPolyMeshDetail* dmesh = rcAllocPolyMeshDetail();
    if (!dmesh) {
        std::cerr << "buildDetourMeshFromTriangles: rcAllocPolyMeshDetail failed" << std::endl;
        rcFreePolyMesh(pmesh);
        rcFreeContourSet(cset);
        rcFreeCompactHeightfield(chf);
        rcFreeHeightField(solid);
        return nullptr;
    }
    if (!rcBuildPolyMeshDetail(&ctx, *pmesh, *chf, cfg.detailSampleDist, cfg.detailSampleMaxError, *dmesh)) {
        std::cerr << "buildDetourMeshFromTriangles: rcBuildPolyMeshDetail failed" << std::endl;
        rcFreePolyMeshDetail(dmesh);
        rcFreePolyMesh(pmesh);
        rcFreeContourSet(cset);
        rcFreeCompactHeightfield(chf);
        rcFreeHeightField(solid);
        return nullptr;
    }

    if (pmesh->nverts == 0 || pmesh->npolys == 0) {
        std::cerr << "buildDetourMeshFromTriangles: empty polymesh (nverts or npolys == 0)" << std::endl;
        rcFreePolyMeshDetail(dmesh);
        rcFreePolyMesh(pmesh);
        rcFreeContourSet(cset);
        rcFreeCompactHeightfield(chf);
        rcFreeHeightField(solid);
        return nullptr;
    }

    unsigned char* navData = nullptr;
    int navDataSize = 0;

    dtNavMeshCreateParams params;
    memset(&params, 0, sizeof(params));
    params.verts = pmesh->verts;
    params.vertCount = pmesh->nverts;
    params.polys = pmesh->polys;
    params.polyAreas = pmesh->areas;
    params.polyFlags = pmesh->flags;
    params.polyCount = pmesh->npolys;
    params.nvp = pmesh->nvp;
    params.detailMeshes = dmesh->meshes;
    params.detailVerts = dmesh->verts;
    params.detailVertsCount = dmesh->nverts;
    params.detailTris = dmesh->tris;
    params.detailTriCount = dmesh->ntris;
    params.walkableHeight = agentHeight;
    params.walkableRadius = agentRadius;
    params.walkableClimb = agentMaxClimb;
    rcVcopy(params.bmin, pmesh->bmin);
    rcVcopy(params.bmax, pmesh->bmax);
    params.cs = cfg.cs;
    params.ch = cfg.ch;
    params.buildBvTree = true;

    bool createOK = false;
    if (dtCreateNavMeshData(&params, &navData, &navDataSize)) {
        createOK = true;
    } else {
        createOK = false;
    }

    if (!createOK || !navData || navDataSize <= 0) {
        std::cerr << "buildDetourMeshFromTriangles: dtCreateNavMeshData failed (createOK=" << createOK
                  << " navData=" << (void*)navData << " navDataSize=" << navDataSize << ")" << std::endl;

        std::cerr << "Dump sample polys (first up to 8 polys):" << std::endl;
        int dumpPolys = std::min(pmesh->npolys, 8);
        for (int pi = 0; pi < dumpPolys; ++pi) {
            unsigned short* poly = &pmesh->polys[pi * pmesh->nvp * 2];
            std::cerr << " poly[" << pi << "] verts:";
            for (int vi = 0; vi < pmesh->nvp; ++vi) {
                int viidx = poly[vi];
                if (viidx == RC_MESH_NULL_IDX) break;
                if (viidx < 0 || viidx >= pmesh->nverts) {
                    std::cerr << " INVALID_IDX(" << viidx << ")";
                } else {
                    std::cerr << " " << viidx;
                }
            }
            std::cerr << std::endl;
        }

        if (navData) dtFree(navData);
        rcFreePolyMeshDetail(dmesh);
        rcFreePolyMesh(pmesh);
        rcFreeContourSet(cset);
        rcFreeCompactHeightfield(chf);
        rcFreeHeightField(solid);
        return nullptr;
    }

    // If requested, export polymesh verts and triangulated faces (fan triangulation per poly)
    if (outVerts) {
        outVerts->clear();
        outVerts->reserve(pmesh->nverts * 3);

        // Handle two common internal storage cases:
        //  - verts stored as float (3 floats per vertex)
        //  - verts stored as unsigned short (3 ushorts per vertex) quantized; convert to world space using pmesh->bmin and cfg.cs
        using VertElemType = decltype(pmesh->verts[0]);
        if (sizeof(VertElemType) == sizeof(float)) {
            // verts are float
            const float* pv = reinterpret_cast<const float*>(pmesh->verts);
            for (int vi = 0; vi < pmesh->nverts; ++vi) {
                outVerts->push_back(pv[vi*3 + 0]);
                outVerts->push_back(pv[vi*3 + 1]);
                outVerts->push_back(pv[vi*3 + 2]);
            }
        } else {
            // assume unsigned short (or other integer type). Convert to world coords.
            const unsigned short* pv = reinterpret_cast<const unsigned short*>(pmesh->verts);
            for (int vi = 0; vi < pmesh->nverts; ++vi) {
                float x = pmesh->bmin[0] + pv[vi*3 + 0] * cfg.cs;
                float y = pmesh->bmin[1] + pv[vi*3 + 1] * cfg.cs;
                float z = pmesh->bmin[2] + pv[vi*3 + 2] * cfg.cs;
                outVerts->push_back(x);
                outVerts->push_back(y);
                outVerts->push_back(z);
            }
        }
    }

    if (outTris) {
        outTris->clear();
        // For each polygon, read its vertex indices and create triangle fan (v0, vj, vj+1)
        for (int pi = 0; pi < pmesh->npolys; ++pi) {
            unsigned short* poly = &pmesh->polys[pi * pmesh->nvp * 2];
            std::vector<int> vids;
            vids.reserve(pmesh->nvp);
            for (int vi = 0; vi < pmesh->nvp; ++vi) {
                int idx = poly[vi];
                if (idx == RC_MESH_NULL_IDX) break;
                if (idx < 0 || idx >= pmesh->nverts) break;
                vids.push_back(idx);
            }
            if ((int)vids.size() < 3) continue;
            for (size_t j = 1; j + 1 < vids.size(); ++j) {
                outTris->push_back(vids[0]);
                outTris->push_back(vids[j]);
                outTris->push_back(vids[j+1]);
            }
        }
    }

    dtNavMesh* nav = dtAllocNavMesh();
    if (!nav) {
        std::cerr << "buildDetourMeshFromTriangles: dtAllocNavMesh failed" << std::endl;
        dtFree(navData);
        rcFreePolyMeshDetail(dmesh);
        rcFreePolyMesh(pmesh);
        rcFreeContourSet(cset);
        rcFreeCompactHeightfield(chf);
        rcFreeHeightField(solid);
        return nullptr;
    }
    if (dtStatusFailed(nav->init(navData, navDataSize, DT_TILE_FREE_DATA))) {
        std::cerr << "buildDetourMeshFromTriangles: nav->init failed" << std::endl;
        dtFree(navData);
        dtFreeNavMesh(nav);
        rcFreePolyMeshDetail(dmesh);
        rcFreePolyMesh(pmesh);
        rcFreeContourSet(cset);
        rcFreeCompactHeightfield(chf);
        rcFreeHeightField(solid);
        return nullptr;
    }

    rcFreePolyMeshDetail(dmesh);
    rcFreePolyMesh(pmesh);
    rcFreeContourSet(cset);
    rcFreeCompactHeightfield(chf);
    rcFreeHeightField(solid);

    return nav;
}

void freeDetourMesh(dtNavMesh* mesh) {
    if (!mesh) return;
    dtFreeNavMesh(mesh);
}