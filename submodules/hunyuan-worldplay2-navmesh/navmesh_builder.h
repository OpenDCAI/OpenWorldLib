#pragma once

#include <vector>
#include <cstdint>

// forward declare Detour type
struct dtNavMesh;

// 构建函数：
// - vertices: 扁平的 float 列表 x0,y0,z0, x1,y1,z1, ...
// - indices:  扁平的 int 列表 三角形索引 0,1,2, 3,4,5, ...
// - outVerts: 可选，若非空则在构建成功时填充为 polymesh 顶点扁平数组 (x,y,z,...)
// - outTris:  可选，若非空则在构建成功时填充为基于 outVerts 的三角形索引扁平数组 (i0,i1,i2,...)
// 返回：如果成功，返回非空的 dtNavMesh*（由调用者负责释放）
//       如果失败，返回 nullptr
// 说明：函数内部会尝试把 Recast 生成的 polymesh（顶点和按多边形扇形三角化的三角形）拷贝到
//       outVerts/outTris（如果对应指针非空）。这样可以在不依赖 dtNavMesh 私有 API 的情况下
//       在 Python 侧取得多边形网格数据。
dtNavMesh* buildDetourMeshFromTriangles(const std::vector<float>& vertices,
                                        const std::vector<int>& indices,
                                        float cellSize = 0.3f,
                                        float cellHeight = 0.2f,
                                        float agentHeight = 2.0f,
                                        float agentRadius = 0.6f,
                                        float agentMaxClimb = 0.9f,
                                        float maxSlope = 45.0f,
                                        std::vector<float>* outVerts = nullptr,
                                        std::vector<int>* outTris = nullptr);
void freeDetourMesh(dtNavMesh* mesh);