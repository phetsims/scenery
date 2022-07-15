// Copyright 2022, University of Colorado Boulder

/**
 * Rich enumeration for internal layout code
 *
 * NOTE: This is orientation-agnostic for a reason, so that it's natural with GridBox, and FlowBox can switch
 * orientation
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Orientation from '../../../phet-core/js/Orientation.js';
import { scenery } from '../imports.js';
import EnumerationValue from '../../../phet-core/js/EnumerationValue.js';
import Enumeration from '../../../phet-core/js/Enumeration.js';

export const HorizontalLayoutAlignValues = [ 'left', 'right', 'center', 'origin' ] as const;
export type HorizontalLayoutAlign = typeof HorizontalLayoutAlignValues[number];

export const VerticalLayoutAlignValues = [ 'top', 'bottom', 'center', 'origin' ] as const;
export type VerticalLayoutAlign = typeof VerticalLayoutAlignValues[number];

export default class LayoutAlign extends EnumerationValue {
  public static readonly START = new LayoutAlign( 'left', 'top', 0 );
  public static readonly END = new LayoutAlign( 'right', 'bottom', 1 );
  public static readonly CENTER = new LayoutAlign( 'center', 'center', 0.5 );
  public static readonly ORIGIN = new LayoutAlign( 'origin', 'origin' );

  // String enumeration types for the horizontal orientation
  public readonly horizontal: HorizontalLayoutAlign;

  // String enumeration types for the vertical orientation
  public readonly vertical: VerticalLayoutAlign;

  // A multiplier value used in the padding computation
  public readonly padRatio: number;

  public constructor( horizontal: HorizontalLayoutAlign, vertical: VerticalLayoutAlign, padRatio: number = Number.POSITIVE_INFINITY ) {
    super();

    this.horizontal = horizontal;
    this.vertical = vertical;
    this.padRatio = padRatio;
  }

  public static readonly enumeration = new Enumeration( LayoutAlign, {
    phetioDocumentation: 'Alignment for layout containers'
  } );

  public static getAllowedAligns( orientation: Orientation ): readonly ( string | null )[] {
    return [ ...( orientation === Orientation.HORIZONTAL ? HorizontalLayoutAlignValues : VerticalLayoutAlignValues ), null ];
  }

  // Converts a string union value into the internal Enumeration value
  public static alignToInternal( orientation: Orientation, key: HorizontalLayoutAlign | VerticalLayoutAlign | null ): LayoutAlign | null {
    return orientation === Orientation.HORIZONTAL
           ? LayoutAlign.horizontalAlignToInternal( key as HorizontalLayoutAlign )
           : LayoutAlign.verticalAlignToInternal( key as VerticalLayoutAlign );
  }

  public static horizontalAlignToInternal( key: HorizontalLayoutAlign | null ): LayoutAlign | null {
    if ( key === null ) {
      return null;
    }

    assert && assert( horizontalAlignMap[ key ] );

    return horizontalAlignMap[ key ];
  }

  public static verticalAlignToInternal( key: VerticalLayoutAlign | null ): LayoutAlign | null {
    if ( key === null ) {
      return null;
    }

    assert && assert( verticalAlignMap[ key ] );

    return verticalAlignMap[ key ];
  }

  // Converts an internal Enumeration value into a string union value.
  public static internalToAlign( orientation: Orientation, align: LayoutAlign | null ): HorizontalLayoutAlign | VerticalLayoutAlign | null {
    if ( align === null ) {
      return null;
    }
    else if ( orientation === Orientation.HORIZONTAL ) {
      return align.horizontal;
    }
    else {
      return align.vertical;
    }
  }
}

const horizontalAlignMap = {
  left: LayoutAlign.START,
  right: LayoutAlign.END,
  center: LayoutAlign.CENTER,
  origin: LayoutAlign.ORIGIN
};
const verticalAlignMap = {
  top: LayoutAlign.START,
  bottom: LayoutAlign.END,
  center: LayoutAlign.CENTER,
  origin: LayoutAlign.ORIGIN

};

scenery.register( 'LayoutAlign', LayoutAlign );
