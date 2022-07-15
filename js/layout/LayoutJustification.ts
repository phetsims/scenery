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

export const HorizontalLayoutJustificationValues = [ 'left', 'right', 'center', 'spaceBetween', 'spaceAround', 'spaceEvenly' ] as const;
export type HorizontalLayoutJustification = typeof HorizontalLayoutJustificationValues[number];

export const VerticalLayoutJustificationValues = [ 'top', 'bottom', 'center', 'spaceBetween', 'spaceAround', 'spaceEvenly' ] as const;
export type VerticalLayoutJustification = typeof VerticalLayoutJustificationValues[number];

// Given an amount of extra space remaining and a lineLength, creates a function that maps the cell index to how much
// spacing (for justification) should be in front of that cell.
type SpaceRemainingFunctionFactory = ( spaceRemaining: number, lineLength: number ) => ( ( index: number ) => number );

export default class LayoutJustification extends EnumerationValue {
  public static readonly START = new LayoutJustification(
    () => () => 0,
    'left', 'top'
  );

  public static readonly END = new LayoutJustification(
    spaceRemaining => index => index === 0 ? spaceRemaining : 0,
    'right', 'bottom'
  );

  public static readonly CENTER = new LayoutJustification(
    spaceRemaining => index => index === 0 ? spaceRemaining / 2 : 0,
    'center', 'center'
  );

  public static readonly SPACE_BETWEEN = new LayoutJustification(
    ( spaceRemaining, lineLength ) => index => index !== 0 ? ( spaceRemaining / ( lineLength - 1 ) ) : 0,
    'spaceBetween', 'spaceBetween'
  );

  public static readonly SPACE_AROUND = new LayoutJustification(
    ( spaceRemaining, lineLength ) => index => ( index !== 0 ? 2 : 1 ) * spaceRemaining / ( 2 * lineLength ),
    'spaceAround', 'spaceAround'
  );

  public static readonly SPACE_EVENLY = new LayoutJustification(
    ( spaceRemaining, lineLength ) => index => spaceRemaining / ( lineLength + 1 ),
    'spaceEvenly', 'spaceEvenly'
  );

  // String enumeration types for the horizontal orientation
  public readonly horizontal: HorizontalLayoutJustification;

  // String enumeration types for the vertical orientation
  public readonly vertical: VerticalLayoutJustification;

  // See SpaceRemainingFunctionFactory for docs
  public readonly spacingFunctionFactory: SpaceRemainingFunctionFactory;

  public constructor( spacingFunctionFactory: SpaceRemainingFunctionFactory, horizontal: HorizontalLayoutJustification, vertical: VerticalLayoutJustification ) {
    super();

    this.spacingFunctionFactory = spacingFunctionFactory;
    this.horizontal = horizontal;
    this.vertical = vertical;
  }

  public static readonly enumeration = new Enumeration( LayoutJustification, {
    phetioDocumentation: 'Justify for layout containers'
  } );

  public static getAllowedJustificationValues( orientation: Orientation ): readonly string[] {
    return orientation === Orientation.HORIZONTAL ? HorizontalLayoutJustificationValues : VerticalLayoutJustificationValues;
  }

  // Converts a string union value into the internal Enumeration value
  public static justifyToInternal( orientation: Orientation, key: HorizontalLayoutJustification | VerticalLayoutJustification ): LayoutJustification {
    if ( orientation === Orientation.HORIZONTAL ) {
      assert && assert( horizontalJustificationMap[ key as 'left' | 'right' | 'center' | 'spaceBetween' | 'spaceAround' | 'spaceEvenly' ] );

      return horizontalJustificationMap[ key as 'left' | 'right' | 'center' | 'spaceBetween' | 'spaceAround' | 'spaceEvenly' ];
    }
    else {
      assert && assert( verticalJustificationMap[ key as 'top' | 'bottom' | 'center' | 'spaceBetween' | 'spaceAround' | 'spaceEvenly' ] );

      return verticalJustificationMap[ key as 'top' | 'bottom' | 'center' | 'spaceBetween' | 'spaceAround' | 'spaceEvenly' ];
    }
  }

  // Converts an internal Enumeration value into a string union value.
  public static internalToJustify( orientation: Orientation, justify: LayoutJustification ): HorizontalLayoutJustification | VerticalLayoutJustification {
    if ( orientation === Orientation.HORIZONTAL ) {
      return justify.horizontal;
    }
    else {
      return justify.vertical;
    }
  }
}

const horizontalJustificationMap = {
  left: LayoutJustification.START,
  right: LayoutJustification.END,
  center: LayoutJustification.CENTER,
  spaceBetween: LayoutJustification.SPACE_BETWEEN,
  spaceAround: LayoutJustification.SPACE_AROUND,
  spaceEvenly: LayoutJustification.SPACE_EVENLY
};
const verticalJustificationMap = {
  top: LayoutJustification.START,
  bottom: LayoutJustification.END,
  center: LayoutJustification.CENTER,
  spaceBetween: LayoutJustification.SPACE_BETWEEN,
  spaceAround: LayoutJustification.SPACE_AROUND,
  spaceEvenly: LayoutJustification.SPACE_EVENLY
};

scenery.register( 'LayoutJustification', LayoutJustification );
