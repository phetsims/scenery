// Copyright 2024, University of Colorado Boulder

/**
 * A property that is true when the node appears on the given display. See DisplayedTrailsProperty for additional options
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { DisplayedTrailsProperty, DisplayedTrailsPropertyOptions, Node, scenery, Trail } from '../imports.js';
import { DerivedProperty1, DerivedPropertyOptions } from '../../../axon/js/DerivedProperty.js';

export type DisplayedPropertyOptions = DisplayedTrailsPropertyOptions & DerivedPropertyOptions<boolean>;

class DisplayedProperty extends DerivedProperty1<boolean, Trail[]> {

  private readonly displayedTrailsProperty: DisplayedTrailsProperty;

  public constructor( node: Node, options?: DisplayedPropertyOptions ) {

    const displayedTrailsProperty = new DisplayedTrailsProperty( node, options );

    super( [ displayedTrailsProperty ], trails => trails.length > 0, options );

    this.displayedTrailsProperty = displayedTrailsProperty;
  }

  public override dispose(): void {
    this.displayedTrailsProperty.dispose();

    super.dispose();
  }
}

scenery.register( 'DisplayedProperty', DisplayedProperty );
export default DisplayedProperty;