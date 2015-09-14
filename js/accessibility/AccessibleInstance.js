// Copyright 2002-2014, University of Colorado Boulder

/**
 * An instance that is synchronously created, for handling accessibility needs.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Events = require( 'AXON/Events' );
  var scenery = require( 'SCENERY/scenery' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );

  var globalId = 1;

  scenery.AccessibleInstance = function AccessibleInstance( display, trail ) {
    this.initializeAccessibleInstance( display, trail );
  };
  var AccessibleInstance = scenery.AccessibleInstance;

  inherit( Events, AccessibleInstance, {
    /**
     * @param {DOMElement} [domElement] - If not included here, subtype is responsible for setting it in the constructor.
     */
    initializeAccessibleInstance: function( display, trail ) {
      Events.call( this ); // TODO: is Events worth mixing in by default? Will we need to listen to events?

      assert && assert( !this.id || this.disposed, 'If we previously existed, we need to have been disposed' );

      // unique ID
      this.id = this.id || globalId++;

      this.display = display;
      this.trail = trail;
      this.node = trail.lastNode();

      this.children = cleanArray( this.children );
      this.node.addAccessibleInstance( this );

      return this;
    },

    /**
     * Consider the following example:
     *
     * We have a node structure:
     * A
     *  B ( accessible )
     *    C (accessible )
     *      D
     *        E (accessible)
     *         G (accessible)
     *        F
     *          H (accessible)
     *
     *
     * Which has an equivalent accessible instance tree:
     * root
     *  AB
     *    ABC
     *      ABCDE
     *        ABCDEG
     *      ABCDFH
     *
     * Produces the call tree for adding instances to the accessible instance tree:
     * ABC.addSubtree( ABCD ) - not accessible
     *     ABC.addSubtree( ABCDE)
     *       ABCDE.addSubtree( ABCDEG )
     *     ABC.addSubtree( ABCDF )
     *       ABC.addSubtree( ABCDFH )
     */
    addSubtree: function( trail ) {
      var node = trail.lastNode();
      var nextInstance = this;
      if ( node.accessibleContent ) {
        var accessibleInstance = new AccessibleInstance( this.display, trail.copy() ); // TODO: Pooling
        this.children.push( accessibleInstance ); // TODO: Mark us as dirty for performance.

        nextInstance = accessibleInstance;
      }
      var children = node._children;
      for ( var i = 0; i < children.length; i++ ) {
        trail.addDescendant( children[ i ], i );
        nextInstance.addSubtree( trail );
        trail.removeDescendant();
      }
    },

    dispose: function() {
      this.node.removeAccessibleInstance( this );
    }
  } );

  return AccessibleInstance;
} );
