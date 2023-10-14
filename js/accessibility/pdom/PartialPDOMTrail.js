// Copyright 2018-2023, University of Colorado Boulder

/**
 * Represents a path up to an PDOMInstance.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../imports.js';

class PartialPDOMTrail {
  /**
   * @param {PDOMInstance} pdomInstance
   * @param {Trail} trail
   * @param {boolean} isRoot
   */
  constructor( pdomInstance, trail, isRoot ) {

    // @public
    this.pdomInstance = pdomInstance;
    this.trail = trail;

    // TODO: remove this, since it can be computed from the pdomInstance https://github.com/phetsims/scenery/issues/1581
    this.isRoot = isRoot;

    // @public {Trail} - a full Trail (rooted at our display) to our trail's final node.
    this.fullTrail = this.pdomInstance.trail.copy();
    // NOTE: Only if the parent instance is the root instance do we want to include our partial trail's root.
    // For other instances, this node in the trail will already be included
    // TODO: add Trail.concat() https://github.com/phetsims/scenery/issues/1581
    for ( let j = ( this.isRoot ? 0 : 1 ); j < this.trail.length; j++ ) {
      this.fullTrail.addDescendant( this.trail.nodes[ j ] );
    }
  }
}

scenery.register( 'PartialPDOMTrail', PartialPDOMTrail );
export default PartialPDOMTrail;