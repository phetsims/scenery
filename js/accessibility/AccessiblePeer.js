// Copyright 2002-2014, University of Colorado Boulder

/**
 * An accessible peer controls the appearance of an accessible Node's instance in the parallel DOM.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var Events = require( 'AXON/Events' );
  var scenery = require( 'SCENERY/scenery' );

  var globalId = 1;

  scenery.AccessiblePeer = function AccessiblePeer( domElement, containerDOMElement ) {
    this.initializeAccessiblePeer( domElement, containerDOMElement );
  };
  var AccessiblePeer = scenery.AccessiblePeer;

  inherit( Events, AccessiblePeer, {
    /**
     * @param {DOMElement} [domElement] - If not included here, subtype is responsible for setting it in the constructor.
     */
    initializeAccessiblePeer: function( domElement, containerDOMElement ) {
      Events.call( this ); // TODO: is Events worth mixing in by default? Will we need to listen to events?

      assert && assert( !this.id || this.disposed, 'If we previously existed, we need to have been disposed' );

      // unique ID for drawables
      this.id = this.id || globalId++;

      this.domElement = domElement ? domElement : ( this.domElement || null );
      this.containerDOMElement = containerDOMElement ? containerDOMElement : ( this.containerDOMElement || null );

      this.disposed = false;

      return this;
    },

    getChildContainerElement: function() {
      return this.containerDOMElement || this.domElement;
    },

    dispose: function() {
      this.disposed = true;

      // for now
      this.freeToPool && this.freeToPool();
    }
  } );

  // TODO: evaluate pooling, and is it OK to pool only some peers?
  AccessiblePeer.Poolable = {
    mixin: function( selfDrawableType ) {
      // for pooling, allow <AccessiblePeerType>.createFromPool( accessibleInstance ) and accessiblePeer.freeToPool().
      // Creation will initialize the peer to an initial state.
      Poolable.mixin( selfDrawableType, {
        defaultFactory: function() {
          /* jshint -W055 */
          return new selfDrawableType();
        },
        constructorDuplicateFactory: function( pool ) {
          return function( accessibleInstance ) {
            if ( pool.length ) {
              return pool.pop().initialize( accessibleInstance );
            }
            else {
              /* jshint -W055 */
              return new selfDrawableType( accessibleInstance );
            }
          };
        }
      } );
    }
  };

  return AccessiblePeer;
} );
