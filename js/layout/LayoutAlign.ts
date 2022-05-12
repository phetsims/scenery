// Copyright 2022, University of Colorado Boulder

/**
 * Rich enumeration for internal layout code
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

// Disable for the whole file
/* eslint-disable no-protected-jsdoc */

import Orientation from '../../../phet-core/js/Orientation.js';
import { scenery } from '../imports.js';
import EnumerationValue from '../../../phet-core/js/EnumerationValue.js';
import Enumeration from '../../../phet-core/js/Enumeration.js';

export const HorizontalLayoutAlignValues = [ 'left', 'right', 'center', 'origin' ] as const;
export type HorizontalLayoutAlign = typeof HorizontalLayoutAlignValues[number];

export const VerticalLayoutAlignValues = [ 'top', 'bottom', 'center', 'origin' ] as const;
export type VerticalLayoutAlign = typeof VerticalLayoutAlignValues[number];

export default class LayoutAlign extends EnumerationValue {
  static readonly START = new LayoutAlign( 'left', 'top', 0 );
  static readonly END = new LayoutAlign( 'right', 'bottom', 1 );
  static readonly CENTER = new LayoutAlign( 'center', 'center', 0.5 );
  static readonly ORIGIN = new LayoutAlign( 'origin', 'origin' );

  readonly horizontal: HorizontalLayoutAlign;
  readonly vertical: VerticalLayoutAlign;
  readonly padRatio: number;

  constructor( horizontal: HorizontalLayoutAlign, vertical: VerticalLayoutAlign, padRatio: number = Number.POSITIVE_INFINITY ) {
    super();

    this.horizontal = horizontal;
    this.vertical = vertical;
    this.padRatio = padRatio;
  }

  static readonly enumeration = new Enumeration( LayoutAlign, {
    phetioDocumentation: 'Alignment for layout containers'
  } );

  static getAllowedAligns( orientation: Orientation ): readonly ( string | null )[] {
    return [ ...( orientation === Orientation.HORIZONTAL ? HorizontalLayoutAlignValues : VerticalLayoutAlignValues ), null ];
  }

  static alignToInternal( orientation: Orientation, key: HorizontalLayoutAlign | VerticalLayoutAlign | null ): LayoutAlign | null {
    return orientation === Orientation.HORIZONTAL
           ? LayoutAlign.horizontalAlignToInternal( key as HorizontalLayoutAlign )
           : LayoutAlign.verticalAlignToInternal( key as VerticalLayoutAlign );
  }

  static horizontalAlignToInternal( key: HorizontalLayoutAlign | null ): LayoutAlign | null {
    if ( key === null ) {
      return null;
    }

    assert && assert( horizontalAlignMap[ key as 'left' | 'right' | 'center' | 'origin' ] );

    return horizontalAlignMap[ key as 'left' | 'right' | 'center' | 'origin' ];
  }

  static verticalAlignToInternal( key: VerticalLayoutAlign | null ): LayoutAlign | null {
    if ( key === null ) {
      return null;
    }

    assert && assert( verticalAlignMap[ key as 'top' | 'bottom' | 'center' | 'origin' ] );

    return verticalAlignMap[ key as 'top' | 'bottom' | 'center' | 'origin' ];
  }

  static internalToAlign( orientation: Orientation, align: LayoutAlign | null ): HorizontalLayoutAlign | VerticalLayoutAlign | null {
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
