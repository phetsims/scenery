// Copyright 2013-2017, University of Colorado Boulder

/**
 * Listens to the fill or stroke of a Node, and notifies when the actual represented value has changed.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var Color = require( 'SCENERY/util/Color' );
  var Gradient = require( 'SCENERY/util/Gradient' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Property = require( 'AXON/Property' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * An observer for a fill or stroke, that will be able to trigger notifications when it changes.
   * @constructor
   *
   * @param {string} type - Either 'fill' or 'stroke'
   * @param {function} changeCallback - To be called on any change (with no arguments)
   */
  function PaintObserver( type, changeCallback ) {
    assert && assert( type === 'fill' || type === 'stroke' );

    // @private {string} - 'fill' or 'stroke'
    this.type = type;

    // @private {string} - Property name on the Node itself
    this.name = '_' + type;

    // @private {function} - Our callback
    this.changeCallback = changeCallback;

    // @private {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern}
    // Our unwrapped fill/stroke value
    this.primary = null;

    // @private {function} - To be called whenever our secondary fill/stroke value may have changed
    this.updateSecondaryListener = this.updateSecondary.bind( this );
  }

  scenery.register( 'PaintObserver', PaintObserver );

  inherit( Object, PaintObserver, {
    /**
     * Initializes our PaintObserver for a specific Node.
     * @public (scenery-internal)
     *
     * @param {Paintable} node
     */
    initialize: function( node ) {
      assert && assert( node !== null );
      this.node = node;

      this.updatePrimary();
    },

    /**
     * Should be called when our paint (fill/stroke) may have changed.
     * @public (scenery-internal)
     *
     * Should update any listeners (if necessary), and call the callback (if necessary)
     */
    updatePrimary: function() {
      var primary = this.node[ this.name ];
      if ( primary !== this.primary ) {
        this.detachPrimary( this.primary );
        this.attachPrimary( primary );
        this.changeCallback();
      }
    },

    /**
     * Called when the value of a "primary" Property (contents of one, main or as a Gradient) is potentially changed.
     * @private
     *
     * @param {string|Color} newPaint
     * @param {string|Color} oldPaint
     */
    updateSecondary: function( newPaint, oldPaint ) {
      this.detachSecondary( oldPaint );
      this.attachSecondary( newPaint );
    },

    /**
     * Attempt to attach listeners to the paint's primary (the paint itself).
     * @private
     *
     * NOTE: If it's a Property, we'll also need to handle the secondary (part inside the Property).
     *
     * @param {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern} paint
     */
    attachPrimary: function( paint ) {
      this.primary = paint;
      if ( paint instanceof Property ) {
        paint.lazyLink( this.updateSecondaryListener );
        this.attachSecondary( paint.get() );
      }
      else if ( paint instanceof Color ) {
        paint.addChangeListener( this.changeCallback );
      }
      else if ( paint instanceof Gradient ) {
        for ( var i = 0; i < paint.stops.length; i++ ) {
          this.attachSecondary( paint.stops[ i ].color );
        }
      }
    },

    /**
     * Attempt to detach listeners from the paint's primary (the paint itself).
     * @private
     *
     * NOTE: If it's a Property or Gradient, we'll also need to handle the secondaries (part(s) inside the Property(ies)).
     *
     * @param {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern} paint
     */
    detachPrimary: function( paint ) {
      if ( paint instanceof Property ) {
        paint.unlink( this.updateSecondaryListener );
        this.detachSecondary( paint.get() );
      }
      else if ( paint instanceof Color ) {
        paint.removeChangeListener( this.changeCallback );
      }
      else if ( paint instanceof Gradient ) {
        for ( var i = 0; i < paint.stops.length; i++ ) {
          this.detachSecondary( paint.stops[ i ].color );
        }
      }
      this.primary = null;
    },

    /**
     * Attempt to attach listeners to the paint's secondary (part within the Property).
     * @private
     *
     * @param {string|Color} paint
     */
    attachSecondary: function( paint ) {
      if ( paint instanceof Color ) {
        paint.addChangeListener( this.changeCallback );
      }
    },

    /**
     * Attempt to detach listeners from the paint's secondary (part within the Property).
     * @private
     *
     * @param {string|Color} paint
     */
    detachSecondary: function( paint ) {
      if ( paint instanceof Color ) {
        paint.removeChangeListener( this.changeCallback );
      }
    },

    /**
     * Cleans our state (so it can be potentially re-used).
     * @public (scenery-internal)
     */
    clean: function() {
      this.detachPrimary( this.primary );
      this.node = null;
    }
  } );

  return PaintObserver;
} );
