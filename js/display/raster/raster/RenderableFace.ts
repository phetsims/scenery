// Copyright 2023, University of Colorado Boulder

/**
 * Represents a face with a RenderProgram/bounds, that can potentially be split into multiples, or optimized
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderExtend, RenderGradientStop, RenderLinearBlend, RenderLinearGradient, RenderProgram, RenderRadialBlend, RenderRadialGradient, scenery } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';

// TODO: naming, omg
export default class RenderableFace {
  public constructor(
    public readonly face: ClippableFace,
    public readonly renderProgram: RenderProgram,
    public readonly bounds: Bounds2
  ) {}

  public static getGradientLinearRanges( min: number, max: number, offset: number, extend: RenderExtend, stops: RenderGradientStop[] ): RenderLinearRange[] {
    const unitRanges: RenderLinearRange[] = [];
    if ( stops[ 0 ].ratio > 0 ) {
      unitRanges.push( new RenderLinearRange(
        0,
        stops[ 0 ].ratio,
        stops[ 0 ].program,
        stops[ 0 ].program
      ) );
    }
    for ( let i = 0; i < stops.length - 1; i++ ) {
      const start = stops[ i ];
      const end = stops[ i + 1 ];

      // If we get a zero-length range, ignore it (it's probably an immediate transition between two colors)
      if ( start.ratio !== end.ratio ) {
        unitRanges.push( new RenderLinearRange(
          start.ratio,
          end.ratio,
          start.program,
          end.program
        ) );
      }
    }
    if ( stops[ stops.length - 1 ].ratio < 1 ) {
      unitRanges.push( new RenderLinearRange(
        stops[ stops.length - 1 ].ratio,
        1,
        stops[ stops.length - 1 ].program,
        stops[ stops.length - 1 ].program
      ) );
    }
    const reversedUnitRanges: RenderLinearRange[] = unitRanges.map( r => r.getUnitReversed() ).reverse();

    // Our used linear ranges
    const linearRanges: RenderLinearRange[] = [];

    const addSectionRanges = ( sectionOffset: number, reversed: boolean ): void => {
      // now relative to our section!
      const sectionMin = min - sectionOffset;
      const sectionMax = max - sectionOffset;

      const ranges = reversed ? reversedUnitRanges : unitRanges;
      ranges.forEach( range => {
        if ( sectionMin < range.end && sectionMax > range.start ) {
          linearRanges.push( range.withOffset( sectionOffset + offset ) );
        }
      } );
    };

    if ( extend === RenderExtend.Pad ) {
      if ( min < 0 ) {
        const firstProgram = stops[ 0 ].program;
        linearRanges.push( new RenderLinearRange(
          Number.NEGATIVE_INFINITY,
          offset,
          firstProgram,
          firstProgram
        ) );
      }
      if ( min <= 1 && max >= 0 ) {
        addSectionRanges( 0, false );
      }
      if ( max > 1 ) {
        const lastProgram = stops[ stops.length - 1 ].program;
        linearRanges.push( new RenderLinearRange(
          1 + offset,
          Number.POSITIVE_INFINITY,
          lastProgram,
          lastProgram
        ) );
      }
    }
    else {
      const isReflect = extend === RenderExtend.Reflect;

      const floorMin = Math.floor( min );
      const ceilMax = Math.ceil( max );

      for ( let i = floorMin; i < ceilMax; i++ ) {
        addSectionRanges( i, isReflect ? i % 2 === 0 : false );
      }
    }

    // Merge adjacent ranges with the same (constant) program
    for ( let i = 0; i < linearRanges.length - 1; i++ ) {
      const range = linearRanges[ i ];
      const nextRange = linearRanges[ i + 1 ];

      if ( range.startProgram === range.endProgram && nextRange.startProgram === nextRange.endProgram && range.startProgram === nextRange.startProgram ) {
        linearRanges.splice( i, 2, new RenderLinearRange(
          range.start,
          nextRange.end,
          range.startProgram,
          range.startProgram
        ) );
        i--;
      }
    }

    return linearRanges;
  }

  public splitLinearGradients(): RenderableFace[] {
    const processedFaces: RenderableFace[] = [];
    const unprocessedFaces: RenderableFace[] = [ this ];

    const findLinearGradient = ( renderProgram: RenderProgram ): RenderLinearGradient | null => {
      let result: RenderLinearGradient | null = null;

      renderProgram.depthFirst( subProgram => {
        // TODO: early exit?
        if ( subProgram instanceof RenderLinearGradient ) {
          result = subProgram;
        }
      } );

      return result;
    };

    while ( unprocessedFaces.length ) {
      const face = unprocessedFaces.pop()!;

      const linearGradient = findLinearGradient( face.renderProgram );

      if ( linearGradient ) {

        const start = linearGradient.transform.timesVector2( linearGradient.start );
        const end = linearGradient.transform.timesVector2( linearGradient.end );

        const delta = end.minus( start );
        const normal = delta.timesScalar( 1 / delta.magnitudeSquared );
        const offset = normal.dot( start );

        // Should evaluate to 1 at the end
        assert && assert( Math.abs( normal.dot( end ) - offset - 1 ) < 1e-8 );

        const dotRange = face.face.getDotRange( normal );

        // relative to gradient "origin"
        const min = dotRange.min - offset;
        const max = dotRange.max - offset;

        const linearRanges = RenderableFace.getGradientLinearRanges( min, max, offset, linearGradient.extend, linearGradient.stops );

        if ( linearRanges.length < 2 ) {
          processedFaces.push( face );
        }
        else {
          const splitValues = linearRanges.map( range => range.start ).slice( 1 );
          const clippedFaces = face.face.getStripeLineClip( normal, splitValues, 0 );

          const renderableFaces = linearRanges.map( ( range, i ) => {
            const clippedFace = clippedFaces[ i ];

            const replacer = ( renderProgram: RenderProgram ): RenderProgram | null => {
              if ( renderProgram !== linearGradient ) {
                return null;
              }

              if ( range.startProgram === range.endProgram ) {
                return range.startProgram.replace( replacer );
              }
              else {
                // We need to rescale our normal for the linear blend, and then adjust our offset to point to the
                // "start":
                // From our original formulation:
                //   normal.dot( startPoint ) = range.start
                //   normal.dot( endPoint ) = range.end
                // with a difference of (range.end - range.start). We want this to be zero, so we rescale our normal:
                //   newNormal.dot( startPoint ) = range.start / ( range.end - range.start )
                //   newNormal.dot( endPoint ) = range.end / ( range.end - range.start )
                // And then we can adjust our offset such that:
                //   newNormal.dot( startPoint ) - offset = 0
                //   newNormal.dot( endPoint ) - offset = 1
                const scaledNormal = normal.timesScalar( 1 / ( range.end - range.start ) );
                const scaledOffset = range.start / ( range.end - range.start );

                return new RenderLinearBlend(
                  null,
                  scaledNormal,
                  scaledOffset,
                  range.startProgram.replace( replacer ),
                  range.endProgram.replace( replacer ),
                  linearGradient.colorSpace
                );
              }
            };

            return new RenderableFace( clippedFace, face.renderProgram.replace( replacer ), clippedFace.getBounds() );
          } ).filter( face => face.face.getArea() > 1e-8 );

          unprocessedFaces.push( ...renderableFaces );
        }
      }
      else {
        processedFaces.push( face );
      }
    }

    return processedFaces;
  }

  public splitRadialGradients(): RenderableFace[] {
    const processedFaces: RenderableFace[] = [];
    const unprocessedFaces: RenderableFace[] = [ this ];

    const findCircularRadialGradient = ( renderProgram: RenderProgram ): RenderRadialGradient | null => {
      let result: RenderRadialGradient | null = null;

      renderProgram.depthFirst( subProgram => {
        // TODO: early exit?
        if ( subProgram instanceof RenderRadialGradient && subProgram.start.equals( subProgram.end ) ) {
          result = subProgram;
        }
      } );

      return result;
    };

    while ( unprocessedFaces.length ) {
      const face = unprocessedFaces.pop()!;

      const radialGradient = findCircularRadialGradient( face.renderProgram );

      if ( radialGradient ) {
        const localClippableFace = face.face.getTransformed( radialGradient.transform.inverted() );

        const center = radialGradient.start;

        const distanceRange = localClippableFace.getDistanceRange( center );

        const isReversed = radialGradient.startRadius > radialGradient.endRadius;

        const minRadius = isReversed ? radialGradient.endRadius : radialGradient.startRadius;
        const maxRadius = isReversed ? radialGradient.startRadius : radialGradient.endRadius;
        const stops = isReversed ? radialGradient.stops.map( stop => {
          return new RenderGradientStop( 1 - stop.ratio, stop.program );
        } ).reverse() : radialGradient.stops;

        const deltaRadius = maxRadius - minRadius;
        const offset = minRadius / deltaRadius;

        const min = ( distanceRange.min / deltaRadius ) - offset;
        const max = ( distanceRange.max / deltaRadius ) - offset;

        const linearRanges = RenderableFace.getGradientLinearRanges( min, max, offset, radialGradient.extend, stops );

        if ( linearRanges.length < 2 ) {
          processedFaces.push( face );
        }
        else {
          const splitValues = linearRanges.map( range => range.start ).slice( 1 );

          // Compute clippedFaces
          const clippedFaces: ClippableFace[] = [];
          let remainingFace = localClippableFace;
          for ( let i = 0; i < splitValues.length; i++ ) {
            const splitValue = splitValues[ i ];

            // TODO: get maxAngleSplit based on magnitude!!!
            const maxAngleSplit = Math.PI / 64;

            const { insideFace, outsideFace } = remainingFace.getBinaryCircularClip( center, splitValue, maxAngleSplit );

            clippedFaces.push( insideFace );
            remainingFace = outsideFace;
          }
          clippedFaces.push( remainingFace );

          const renderableFaces = linearRanges.map( ( range, i ) => {
            const clippedFace = clippedFaces[ i ];

            const replacer = ( renderProgram: RenderProgram ): RenderProgram | null => {
              if ( renderProgram !== radialGradient ) {
                return null;
              }

              if ( range.startProgram === range.endProgram ) {
                return range.startProgram.replace( replacer );
              }
              else {
                const startRadius = minRadius + range.start * deltaRadius;
                const endRadius = minRadius + range.end * deltaRadius;

                return new RenderRadialBlend(
                  null,
                  radialGradient.transform,
                  startRadius,
                  endRadius,
                  range.startProgram.replace( replacer ),
                  range.endProgram.replace( replacer ),
                  radialGradient.colorSpace
                );
              }
            };

            return new RenderableFace( clippedFace, face.renderProgram.replace( replacer ), clippedFace.getBounds() );
          } ).filter( face => face.face.getArea() > 1e-8 );

          unprocessedFaces.push( ...renderableFaces );
        }
      }
      else {
        processedFaces.push( face );
      }
    }

    return processedFaces;
  }
}

class RenderLinearRange {
  public constructor(
    public readonly start: number,
    public readonly end: number,

    // NOTE: equal start/end programs means constant!
    public readonly startProgram: RenderProgram,
    public readonly endProgram: RenderProgram
  ) {}

  public getUnitReversed(): RenderLinearRange {
    return new RenderLinearRange( 1 - this.end, 1 - this.start, this.endProgram, this.startProgram );
  }

  public withOffset( offset: number ): RenderLinearRange {
    return new RenderLinearRange( this.start + offset, this.end + offset, this.startProgram, this.endProgram );
  }
}

scenery.register( 'RenderableFace', RenderableFace );
