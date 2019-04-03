// Copyright 2015-2016, University of Colorado Boulder

/**
 * Used for identifying when any ancestor transform of a node in a trail causes that node's global transform to change.
 * It also provides fast computation of that global matrix, NOT recomputing every matrix, even on most transform
 * changes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Jesse Greenberg
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Matrix3 = require( 'DOT/Matrix3' );

  var scenery = require( 'SCENERY/scenery' );

  /**
   * Creates a transform-tracking object, where it can send out updates on transform changes, and also efficiently
   * compute the transform.
   * @constructor
   * @public
   *
   * @param {Trail} trail
   * @param {Object} [options]
   */
  function TransformTracker( trail, options ) {
    var self = this;

    options = _.extend( {
      isStatic: false // {boolean} - Whether the bounds listeners should be added with on() or onStatic().
    }, options );
    this._isStatic = options.isStatic;

    // @public {Trail}
    this.trail = trail;

    // @private {Array.<Matrix3>|null}
    // this._matrices[ i ] will be equal to: trail.nodes[ 1 ].matrix * ... * trail.nodes[ i + 1 ].matrix
    //
    this._matrices = null; // Will be initialized on first need.

    // @private {number} - this._matrices[ i ] where i >= this._dirtyIndex will need to be recomputed
    this._dirtyIndex = 0;

    // @private {Array.<Function>} - Listeners added by client, will be called on transform changes.
    this._listeners = [];

    // Hook up listeners to each Node in the trail, so we are notified of changes. Will be removed on disposal.
    this._nodeTransformListeners = [];
    for ( var j = 1; j < this.trail.length; j++ ) {
      // Wrapping with closure to prevent changes
      var nodeTransformListener = (function( index ) {
        return function() {
          self.onTransformChange( index );
        };
      })( j - 1 );

      this._nodeTransformListeners.push( nodeTransformListener );

      if ( this._isStatic ) {
        trail.nodes[ j ].onStatic( 'transform', nodeTransformListener );
      }
      else {
        trail.nodes[ j ].on( 'transform', nodeTransformListener );
      }
    }
  }

  scenery.register( 'TransformTracker', TransformTracker );

  inherit( Object, TransformTracker, {
    /**
     * Gets rid of all external references and listeners. This object is inoperable afterwards.
     * @public
     */
    dispose: function() {
      for ( var j = 1; j < this.trail.length; j++ ) {
        var nodeTransformListener = this._nodeTransformListeners[ j - 1 ];

        if ( this._isStatic ) {
          this.trail.nodes[ j ].offStatic( 'transform', nodeTransformListener, false );
        }
        else {
          this.trail.nodes[ j ].off( 'transform', nodeTransformListener, false );
        }
      }
    },

    /**
     * Adds a listener function that will be synchronously called whenever the transform for this Trail changes.
     * @public
     *
     * @param {Function} listener - Listener will be called with no arguments.
     */
    addListener: function( listener ) {
      assert && assert( typeof listener === 'function' );

      this._listeners.push( listener );
    },

    /**
     * Removes a listener that was previously added with addListener().
     * @public
     *
     * @param {Function} listener
     */
    removeListener: function( listener ) {
      assert && assert( typeof listener === 'function' );

      var index = _.indexOf( this._listeners, listener );
      assert && assert( index >= 0, 'TransformTracker listener not found' );

      this._listeners.splice( index, 1 );
    },

    /**
     * Notifies listeners of a transform change.
     * @private
     */
    notifyListeners: function() {
      var listeners = this._listeners;

      if ( !this._isStatic ) {
        listeners = listeners.slice();
      }

      var length = listeners.length;
      for ( var i = 0; i < length; i++ ) {
        listeners[ i ]();
      }
    },

    /**
     * Called when one of the nodes' transforms is changed.
     * @private
     *
     * @param {number} matrixIndex - The index into our matrices array, e.g. this._matrices[ matrixIndex ].
     */
    onTransformChange: function( matrixIndex ) {
      this._dirtyIndex = Math.min( this._dirtyIndex, matrixIndex );
      this.notifyListeners();
    },

    /**
     * Returns the local-to-global transformation matrix for the Trail, which transforms its leaf node's local
     * coordinate frame into the global coordinate frame.
     * @public
     *
     * NOTE: The matrix returned should not be mutated. Please make a copy if needed.
     *
     * @returns {Matrix3}
     */
    getMatrix: function() {
      if ( this._matrices === null ) {
        this._matrices = [];

        // Start at 1, so that we don't include the root node's transform
        for ( var i = 1; i < this.trail.length; i++ ) {
          this._matrices.push( new Matrix3() );
        }
      }

      // If the trail isn't long enough to have a transform, return the identity matrix
      if ( this._matrices.length <= 0 ) {
        return Matrix3.IDENTITY;
      }

      // Starting at the dirty index, recompute matrices.
      var numMatrices = this._matrices.length;
      for ( var index = this._dirtyIndex; index < numMatrices; index++ ) {
        var nodeMatrix = this.trail.nodes[ index + 1 ].matrix;

        if ( index === 0 ) {
          this._matrices[ index ].set( nodeMatrix );
        }
        else {
          this._matrices[ index ].set( this._matrices[ index - 1 ] );
          this._matrices[ index ].multiplyMatrix( nodeMatrix );
        }
      }

      // Reset the dirty index to mark all matrices as 'clean'.
      this._dirtyIndex = numMatrices;

      // Return the last matrix, which contains our composite transformation.
      return this._matrices[ numMatrices - 1 ];
    },
    get matrix() { return this.getMatrix(); }
  } );

  return TransformTracker;
} );
