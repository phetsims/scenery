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

  // Will combine faces that have equivalent RenderPrograms IFF they border each other (leaving separate programs with
  // equivalent RenderPrograms separate if they don't border). It will also remove edges that border between faces
  // that we combine, and will connect edges to keep things polygonal!
  public static toTracedRenderableFaces(
    faces: RationalFace[],
    fromIntegerMatrix: Matrix3
  ): RenderableFace[] {

    // TODO: add algorithm description docs!!! This one is important

    const renderableFaces: RenderableFace[] = [];

    for ( let i = 0; i < faces.length; i++ ) {
      const startingFace = faces[ i ];

      if ( !startingFace.processed ) {
        startingFace.processed = true;

        const polygons: Vector2[][] = []; // A list of polygons we'll append into
        const edges: RationalHalfEdge[] = [
          ...startingFace.getEdges() // defensive copy, could remove sometime
        ]; // A list of edges to process
        const bounds = startingFace.getBounds( fromIntegerMatrix ).copy(); // we're going to mutate this
        const renderProgram = startingFace.renderProgram!;
        assert && assert( renderProgram );

        // Effectively, we cache whether faces or compatible or not
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
              compatibleFaces.add( candidateFace );
              bounds.includeBounds( candidateFace.getBounds( fromIntegerMatrix ) );
              edges.push( ...candidateFace.getEdges() );
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
            traceSimplifier.reset();

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
