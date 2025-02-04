// Copyright 2024-2025, University of Colorado Boulder

/**
 * A property that is true when the node appears on the given display. See DisplayedTrailsProperty for additional options
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { DerivedProperty1, DerivedPropertyOptions } from '../../../axon/js/DerivedProperty.js';
import type { DisplayedTrailsPropertyOptions } from '../util/DisplayedTrailsProperty.js';
import DisplayedTrailsProperty from '../util/DisplayedTrailsProperty.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';
import Trail from '../util/Trail.js';

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