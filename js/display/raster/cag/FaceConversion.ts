// Copyright 2023, University of Colorado Boulder

/**
 * Multiple methods of conversion from RationalFaces to RenderableFaces.
 *
 * They mostly differ on whether they combine faces with equivalent RenderPrograms, WHICH cases they do so, and
 * whether they output polygonal or unsorted-edge formats (PolygonalFace/EdgedFace).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { EdgedFace, LinearEdge, PolygonalFace, RationalFace, RenderableFace, RenderProgram, scenery } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';

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

    return RationalFace.traceCombineFaces(
      faces,
      fromIntegerMatrix,
      ( face: RationalFace ): RenderProgram => face.renderProgram!,
      ( face: PolygonalFace, renderProgram: RenderProgram, bounds: Bounds2 ) => new RenderableFace( face, renderProgram, bounds ),
      ( programA: RenderProgram, programB: RenderProgram | null ) => {
        return !!programB && programA.equals( programB );
      }
    );
  }
}

scenery.register( 'FaceConversion', FaceConversion );
