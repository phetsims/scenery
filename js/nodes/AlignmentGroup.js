// Copyright 2016-2016, University of Colorado Boulder

/**
 * A group of containers that follow the constraints:
 * 1. Every container will have the same bounds, with an upper-left of (0,0)
 * 2. The container sizes will be the smallest possible to fit every container's content (with respective padding).
 * 3. Each container is responsible for positioning its content in its bounds (with customizable alignment and padding).
 *
 * Containers can be dynamically created and disposed, and only active containers will be considered for the bounds.
 *
 * NOTE: Container resizes may not happen immediately, and may be delayed until bounds of a container's child occurs.
 *       layout updates can be forced with group.updateLayout(). If the container's content that changed is connected
 *       to a Scenery display, its bounds will update when Display.updateDisplay() will called, so this will guarantee
 *       that the layout will be applied before it is displayed.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var arrayRemove = require( 'PHET_CORE/arrayRemove' );
  var scenery = require( 'SCENERY/scenery' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Node = require( 'SCENERY/nodes/Node' );
  var AlignmentContainer = require( 'SCENERY/nodes/AlignmentContainer' );

  /**
   * Creates an alignment group that can be composed of multiple containers.
   * @constructor
   * @public
   *
   * Use createContainer() to create containers. You can dispose() individual containers, or call dispose() on this
   * group to dispose all of them.
   */
  function AlignmentGroup() {
    // @private {Array.<AlignmentContainer>}
    this._containers = [];

    // @private {boolean} - Gets locked when certain layout is performed.
    this._resizeLock = false;
  }

  scenery.register( 'AlignmentGroup', AlignmentGroup );

  inherit( Object, AlignmentGroup, {
    /**
     * Creates a container with the given content and options.
     * @public
     *
     * @param {Node} content - Note that the content may be repositioned into place.
     * @param {Object} [options] - See AlignmentContainer's constructor below for specific alignment options.
     *                             Will also be passed to the node's constructor.
     */
    createContainer: function( content, options ) {
      assert && assert( content instanceof Node );

      // Setting the group should call our addContainer()
      return new AlignmentContainer( content, _.extend( {
        group: this
      }, options ) );
    },

    /**
     * Dispose all of the containers.
     * @public
     */
    dispose: function() {
      for ( var i = this._containers.length - 1; i >= 0; i-- ) {
        this._containers[ i ].dispose();
      }
    },

    /**
     * Updates the localBounds and alignment for each container.
     * @public
     *
     * NOTE: Calling this will usually not be necessary outside of Scenery, but this WILL trigger bounds revalidation
     *       for every container, which can force the layout code to run.
     */
    updateLayout: function() {
      if ( this._resizeLock ) { return; }
      this._resizeLock = true;

      // Compute the maximum dimension of our containers' content
      var maxWidth = 0;
      var maxHeight = 0;
      for ( var i = 0; i < this._containers.length; i++ ) {
        var container = this._containers[ i ];

        var bounds = container.getContentBounds();

        // Ignore bad bounds
        if ( bounds.isEmpty() || !bounds.isFinite() ) {
          continue;
        }

        maxWidth = Math.max( maxWidth, bounds.width );
        maxHeight = Math.max( maxHeight, bounds.height );
      }


      if ( maxWidth > 0 && maxHeight > 0 ) {
        var alignBounds = new Bounds2( 0, 0, maxWidth, maxHeight );
        // Apply that maximum dimension for each container
        for ( i = 0; i < this._containers.length; i++ ) {
          this._containers[ i ].alignBounds = alignBounds;
        }
      }

      this._resizeLock = false;
    },

    /**
     * Lets the group know that the container has had its content resized.
     * @private
     *
     * @param {AlignmentContainer}
     */
    onContainerContentResized: function( container ) {
      // TODO: in the future, we could only update this specific container if the others don't need updating.
      this.updateLayout();
    },

    /**
     * Adds the container to the group
     * @private
     */
    addContainer: function( container ) {
      this._containers.push( container );

      // Trigger an update when a container is added
      this.updateLayout();
    },

    /**
     * Removes the container from the group
     * @private
     */
    removeContainer: function( container ) {
      arrayRemove( this._containers, container );

      // Trigger an update when a container is removed
      this.updateLayout();
    }
  } );

  return AlignmentGroup;
} );
