// Copyright 2002-2013, University of Colorado

/**
 * An instance that is specific to the display (not necessarily a global instance, could be in a Canvas cache, etc),
 * that is needed to tracking instance-specific display information, and signals to the display system when other
 * changes are necessary.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  var globalIdCounter = 1;

  /**
   * @param {DisplayInstance|null} parent
   */
  scenery.DisplayInstance = function DisplayInstance( trail ) {
    this.id = globalIdCounter++;
    this.trail = trail;
    this.parent = null; // will be set as needed
    this.children = [];
    this.proxyChild = null;
    this.proxyParent = null;
    this.state = null; // filled in with rendering state later
    this.renderer = null; // filled in later

    // references into the linked list of effectively painted instances (null if nothing is effectively painted under this, both self if we are effectively painted)
    this.firstPainted = null;
    this.lastPainted = null;

    // basically, our linked list of effectively painted instances
    this.nextPainted = null;
    this.previousPainted = null;
  };
  var DisplayInstance = scenery.DisplayInstance;

  inherit( Object, DisplayInstance, {
    appendInstance: function( instance ) {
      this.children.push( instance );
    },

    // since backbone/canvas caches can create stub instances that are effectively painted
    isEffectivelyPainted: function() {
      return this.renderer !== null;
    }
  } );

  return DisplayInstance;
} );
