// Copyright 2023, University of Colorado Boulder

/**
 * Multiple methods of conversion from RationalFaces to RenderableFaces.
 *
 * They mostly differ on whether they combine faces with equivalent RenderPrograms, WHICH cases they do so, and
 * whether they output polygonal or unsorted-edge formats (PolygonalFace/EdgedFace).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClipSimplifier, EdgedFace, LinearEdge, PolygonalFace, RationalFace, RationalHalfEdge, RenderableFace, RenderProgram, scenery } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';

const traceSimplifier = new ClipSimplifier();

class AccumulatingFace {
  public faces = new Set<RationalFace>();
  public facesToProcess: RationalFace[] = [];
  public renderProgram: RenderProgram | null = null;
  public bounds: Bounds2 = Bounds2.NOTHING.copy();
  public clippedEdges: LinearEdge[] = [];
}

export default class FaceConversion {
  public static toPolygonalRenderableFaces(
    faces: RationalFace[],
    fromIntegerMatrix: Matrix3
  ): RenderableFace[] {

    // TODO: naming with above!!
    const renderableFaces: RenderableFace[] = [];
    for ( let i = 0; i < faces.length; i++ ) {
      const face = faces[ i ];
      renderableFaces.push( new RenderableFace(
        face.toPolygonalFace( fromIntegerMatrix ),
        face.renderProgram!,
        face.getBounds( fromIntegerMatrix )
      ) );
    }
    return renderableFaces;
  }

  public static toEdgedRenderableFaces(
    faces: RationalFace[],
    fromIntegerMatrix: Matrix3
  ): RenderableFace[] {

    // TODO: naming with above!!
    const renderableFaces: RenderableFace[] = [];
    for ( let i = 0; i < faces.length; i++ ) {
      const face = faces[ i ];
      renderableFaces.push( new RenderableFace(
        face.toEdgedFace( fromIntegerMatrix ),
        face.renderProgram!,
        face.getBounds( fromIntegerMatrix )
      ) );
    }
    return renderableFaces;
  }

  public static toFullyCombinedRenderableFaces(
    faces: RationalFace[],
    fromIntegerMatrix: Matrix3
  ): RenderableFace[] {

    const faceEquivalenceClasses: Set<RationalFace>[] = [];

    for ( let i = 0; i < faces.length; i++ ) {
      const face = faces[ i ];
      let found = false;

      for ( let j = 0; j < faceEquivalenceClasses.length; j++ ) {
        const faceEquivalenceClass = faceEquivalenceClasses[ j ];
        const representative: RationalFace = faceEquivalenceClass.values().next().value;
        if ( face.renderProgram!.equals( representative.renderProgram! ) ) {
          faceEquivalenceClass.add( face );
          found = true;
          break;
        }
      }

      if ( !found ) {
        const newSet = new Set<RationalFace>();
        newSet.add( face );
        faceEquivalenceClasses.push( newSet );
      }
    }

    const renderableFaces: RenderableFace[] = [];
    for ( let i = 0; i < faceEquivalenceClasses.length; i++ ) {
      const faces = faceEquivalenceClasses[ i ];

      const clippedEdges: LinearEdge[] = [];
      let renderProgram: RenderProgram | null = null;
      const bounds = Bounds2.NOTHING.copy();

      for ( const face of faces ) {
        renderProgram = face.renderProgram!;
        bounds.includeBounds( face.getBounds( fromIntegerMatrix ) );

        for ( const boundary of [
          face.boundary,
          ...face.holes
        ] ) {
          for ( const edge of boundary.edges ) {
            if ( !faces.has( edge.reversed.face! ) ) {
              clippedEdges.push( new LinearEdge(
                fromIntegerMatrix.timesVector2( edge.p0float ),
                fromIntegerMatrix.timesVector2( edge.p1float )
              ) );
            }
          }
        }
      }

      renderableFaces.push( new RenderableFace( new EdgedFace( clippedEdges ), renderProgram!, bounds ) );
    }

    return renderableFaces;
  }

  // Will combine faces that have equivalent RenderPrograms IFF they border each other (leaving separate programs with
  // equivalent RenderPrograms separate if they don't border). It will also remove edges that border between faces
  // that we combine (thus switching to EdgedFaces with unsorted edges).
  public static toSimplifyingCombinedRenderableFaces(
    faces: RationalFace[],
    fromIntegerMatrix: Matrix3
  ): RenderableFace[] {

    const accumulatedFaces: AccumulatingFace[] = [];

    // TODO: see if we need micro-optimizations here
    faces.forEach( face => {
      if ( accumulatedFaces.every( accumulatedFace => !accumulatedFace.faces.has( face ) ) ) {
        const newAccumulatedFace = new AccumulatingFace();
        newAccumulatedFace.faces.add( face );
        newAccumulatedFace.facesToProcess.push( face );
        newAccumulatedFace.renderProgram = face.renderProgram!;
        newAccumulatedFace.bounds.includeBounds( face.getBounds( fromIntegerMatrix ) );

        const incompatibleFaces = new Set<RationalFace>();

        // NOTE: side effects!
        const isFaceCompatible = ( face: RationalFace ): boolean => {
          if ( incompatibleFaces.has( face ) ) {
            return false;
          }
          if ( newAccumulatedFace.faces.has( face ) ) {
            return true;
          }

          // Not in either place, we need to test
          if ( face.renderProgram && newAccumulatedFace.renderProgram!.equals( face.renderProgram ) ) {
            newAccumulatedFace.faces.add( face );
            newAccumulatedFace.facesToProcess.push( face );
            newAccumulatedFace.bounds.includeBounds( face.getBounds( fromIntegerMatrix ) );
            return true;
          }
          else {
            incompatibleFaces.add( face );
            return false;
          }
        };

        accumulatedFaces.push( newAccumulatedFace );

        while ( newAccumulatedFace.facesToProcess.length ) {
          const faceToProcess = newAccumulatedFace.facesToProcess.pop()!;

          for ( const boundary of [
            faceToProcess.boundary,
            ...faceToProcess.holes
          ] ) {
            for ( const edge of boundary.edges ) {
              if ( !isFaceCompatible( edge.reversed.face! ) ) {
                newAccumulatedFace.clippedEdges.push( new LinearEdge(
                  fromIntegerMatrix.timesVector2( edge.p0float ),
                  fromIntegerMatrix.timesVector2( edge.p1float )
                ) );
              }
            }
          }
        }
      }
    } );

    return accumulatedFaces.map( accumulatedFace => new RenderableFace(
      new EdgedFace( accumulatedFace.clippedEdges ),
      accumulatedFace.renderProgram!,
      accumulatedFace.bounds
    ) );
  }

  /**
   * Combines faces that have equivalent RenderPrograms IFF they border each other (leaving separate programs with
   * equivalent RenderPrograms separate if they don't border). It will also remove edges that border between faces
   * that we combine, and will connect edges to keep things polygonal!
   */
  public static toTracedRenderableFaces(
    faces: RationalFace[],
    fromIntegerMatrix: Matrix3
  ): RenderableFace[] {

    // In summary, we'll find an edge between incompatible faces, and then we'll trace that edge (staying only on edges
    // between incompatible faces) until we get back to the starting edge. Once we've done this, we have constructed one
    // polygon.
    //
    // For this algorithm, we mark faces and edges as processed when we've already handled their contribution (or marked
    // them to contribute).
    //
    // We'll naturally run into "compatible" faces as we trace the edges. Whenever we run into a new compatible face,
    // (in isFaceCompatible), we'll add its edges to the pool of edges to trace for this RenderableFace.
    //
    // When we trace edges, we'll always start with one between "incompatible" faces. If its nextEdge is between
    // compatible faces, we can skip it (and use the winged edge data structure to try the "next" edge). Thus we'll
    // skip visiting edges between compatible faces. IF they are all compatible, we'll end back up to our starting
    // edge's reversed edge, and we'll essentially trace backward. If this happens, our simplifier will remove the
    // evidence of this. So it should handle "degenerate" cases (e.g. same face on both sides, or a vertex with only
    // one incident edge) just fine.
    //
    // ADDITIONALLY, we'll only be processing non-fully-transparent faces (they should have been sorted out earlier),
    // so nothing will be compatible with the "unbounded" face.

    const renderableFaces: RenderableFace[] = [];

    for ( let i = 0; i < faces.length; i++ ) {
      const startingFace = faces[ i ];

      if ( !startingFace.processed ) {
        startingFace.processed = true;

        // A list of polygons we'll append into (for our RenderableFace).
        const polygons: Vector2[][] = [];

        // A list of edges remaining to process. NOTE: some of these may be marked as "processed", we will just ignore
        // those. Any time we run across a new compatible face, we'll dump its edges in here.
        const edges: RationalHalfEdge[] = [
          ...startingFace.getEdges() // defensive copy, could remove sometime
        ];

        // We'll need to pass bounds to the RenderableFace constructor, we'll accumulate them here.
        const bounds = startingFace.getBounds( fromIntegerMatrix ).copy(); // we'll mutate this

        // All RenderPrograms should be equivalent, so we'll just use the first one
        const renderProgram = startingFace.renderProgram!;
        assert && assert( renderProgram );

        // Cache whether faces or compatible or not
        const compatibleFaces = new Set<RationalFace>();
        const incompatibleFaces = new Set<RationalFace>();
        compatibleFaces.add( startingFace );

        // NOTE: side effects!
        const isFaceCompatible = ( candidateFace: RationalFace ): boolean => {
          if ( compatibleFaces.has( candidateFace ) ) {
            return true;
          }
          else if ( incompatibleFaces.has( candidateFace ) ) {
            return false;
          }
          else {
            // Not in either place, we need to test (also, the unbounded face won't have a RenderProgram)
            if ( candidateFace.renderProgram && candidateFace.renderProgram.equals( renderProgram ) ) {
              // ADD it to the current renderable face
              assert && assert( !candidateFace.processed, 'We should have already found this' );
              candidateFace.processed = true;
              bounds.includeBounds( candidateFace.getBounds( fromIntegerMatrix ) );
              edges.push( ...candidateFace.getEdges() );

              compatibleFaces.add( candidateFace );
              return true;
            }
            else {
              incompatibleFaces.add( candidateFace );
              return false;
            }
          }
        };

        // We'll have edges appended sometimes in isFaceCompatible() checks.
        while ( edges.length ) {
          const startingEdge = edges.pop()!;

          // If the edge is processed OR both faces are compatible, we'll just skip it anyway. We don't want to start
          // tracing on a loop that we will completely remove
          if ( !startingEdge.processed && ( !isFaceCompatible( startingEdge.face! ) || !isFaceCompatible( startingEdge.reversed.face! ) ) ) {
            // Start an edge trace!

            // We'll use the simplifier to remove duplicate or walked-back points.
            // TODO: check to see if removing arbitrary collinear points helps us a lot here. It might be good, but
            // TODO: we don't want to introduce a lot of error. Probably is additional cost

            // Add the first edge
            let currentEdge = startingEdge;
            traceSimplifier.addTransformed( fromIntegerMatrix, currentEdge.p0float.x, currentEdge.p0float.y );
            currentEdge.processed = true;

            do {
              // Walk edges
              let nextEdge = currentEdge.nextEdge!;
              assert && assert( nextEdge.face === currentEdge.face );

              // BUT don't walk edges that are between compatible faces. Instead, wind around until we find either
              // an incompatible face, OR we walk back our edge (the simplifier will take care of removing this)
              while ( nextEdge !== currentEdge.reversed && isFaceCompatible( nextEdge.reversed.face! ) ) {
                nextEdge = nextEdge.reversed.nextEdge!;
              }

              assert && assert( isFaceCompatible( nextEdge.face! ) || isFaceCompatible( nextEdge.reversed.face! ) );

              // Add subsequent edges
              currentEdge = nextEdge;
              traceSimplifier.addTransformed( fromIntegerMatrix, currentEdge.p0float.x, currentEdge.p0float.y );
              currentEdge.processed = true;
            } while ( currentEdge !== startingEdge );

            const polygon = traceSimplifier.finalize();
            if ( polygon.length >= 3 ) {
              polygons.push( polygon );
            }
          }
        }

        renderableFaces.push( new RenderableFace(
          new PolygonalFace( polygons ),
          startingFace.renderProgram!,
          bounds
        ) );
      }
    }

    return renderableFaces;
  }
}

scenery.register( 'FaceConversion', FaceConversion );
