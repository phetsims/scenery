// Copyright 2023, University of Colorado Boulder

/**
 * Represents a section of a gradient's "color stop" space.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { RenderExtend, RenderGradientStop, RenderProgram, scenery } from '../../../imports.js';

export default class RenderLinearRange {
  public constructor(
    // TODO: doc
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
}

scenery.register( 'RenderLinearRange', RenderLinearRange );
