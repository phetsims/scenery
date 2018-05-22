// Copyright 2018, University of Colorado Boulder

/**
 * Represents a path up to an AccessibleInstance.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * @constructor
   * @param {AccessibleInstance} accessibleInstance
   * @param {Trail} trail
   * @param {boolean} isRoot
   */
  function PartialAccessibleTrail( accessibleInstance, trail, isRoot ) {

    // @public
    this.accessibleInstance = accessibleInstance;
    this.trail = trail;

    // TODO: remove this, since it can be computed from the accessibleInstance
    this.isRoot = isRoot;

    // @public {Trail} - a full Trail (rooted at our display) to our trail's final node.
    this.fullTrail = this.accessibleInstance.trail.copy();
    // NOTE: Only if the parent instance is the root instance do we want to include our partial trail's root.
    // For other instances, this node in the trail will already be included
    // TODO: add Trail.concat()
    for ( var j = ( this.isRoot ? 0 : 1 ); j < this.trail.length; j++ ) {
      this.fullTrail.addDescendant( this.trail.nodes[ j ] );
    }
  }

  scenery.register( 'PartialAccessibleTrail', PartialAccessibleTrail );

  return inherit( Object, PartialAccessibleTrail );
} );
