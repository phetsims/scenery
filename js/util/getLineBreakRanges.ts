// Copyright 2022, University of Colorado Boulder

/**
 * Returns where possible line breaks can exist in a given string, according to the
 * Unicode Line Breaking Algorithm (UAX #14). Uses https://github.com/foliojs/linebreak.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
import { scenery } from '../imports.js';
import optionize from '../../../phet-core/js/optionize.js';
import Range from '../../../dot/js/Range.js';

export type GetLineBreaksOptions = {
  // Line breaks can be "required" or "optional". If this is true, ranges will only be given for required line breaks.
  requiredOnly?: boolean;
};

/**
 * Returns an array of ranges that each cover a section of the string where they should not be split by line breaks.
 * These ranges may exclude things like whitespace in-between words, so if a line is being used, the ranges included
 * should just use the starting-min and ending-max to determine what should be included.
 */
const getLineBreakRanges = ( str: string, providedOptions?: GetLineBreaksOptions ): Range[] => {
  const options = optionize<GetLineBreaksOptions>()( {
    requiredOnly: false
  }, providedOptions );

  const ranges: Range[] = [];

  const lineBreaker = new LineBreaker( str );

  // Make it iterable (this was refactored out, but the typing was awkward)
  lineBreaker[ Symbol.iterator ] = () => {
    return {
      next() {
        const value = lineBreaker.nextBreak();
        if ( value !== null ) {
          return { value: value, done: false };
        }
        else {
          return { done: true };
        }
      }
    };
  };

  let lastIndex = 0;
  for ( const brk of lineBreaker ) {
    const index = brk.position;

    if ( options.requiredOnly && !brk.required ) {
      continue;
    }

    // Don't include empty ranges, if they occur.
    if ( lastIndex !== index ) {
      ranges.push( new Range( lastIndex, index ) );
    }

    lastIndex = brk.position;
  }

  // Ending range, if it's not empty
  if ( lastIndex < str.length ) {
    ranges.push( new Range( lastIndex, str.length ) );
  }

  return ranges;
};

scenery.register( 'getLineBreakRanges', getLineBreakRanges );

export default getLineBreakRanges;
