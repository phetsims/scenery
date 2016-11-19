// Copyright 2013-2016, University of Colorado Boulder

/**
 * Listens to the fill or stroke of a Node, and notifies when the actual represented value has changed.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Color = require( 'SCENERY/util/Color' );
  var Property = require( 'AXON/Property' );

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

    // @private {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern}
    // If our primary is a Property of a Color, we'll also need to attach listeners to the Color
    this.secondary = null;

    // @private {function} - To be called whenever our fill/stroke value may have changed
    this.updateListener = this.update.bind( this );
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

      this.update();
    },

    /**
     * Should be called when our paint (fill/stroke) may have changed.
     * @public (scenery-internal)
     *
     * Should update any listeners (if necessary), and call the callback (if necessary)
     */
    update: function() {
      var primary = this.node[ this.name ];
      if ( primary !== this.primary ) {
        this.detachPrimary( this.primary );
        this.attachPrimary( primary );
        this.changeCallback();
      }
      else if ( primary instanceof Property ) {
        var secondary = primary.get();
        if ( secondary !== this.secondary ) {
          this.detachSecondary( this.secondary );
          this.attachSecondary( secondary );
          this.changeCallback();
        }
      }
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
        paint.lazyLink( this.updateListener );
        this.attachSecondary( paint.get() );
      }
      else if ( paint instanceof Color ) {
        paint.addChangeListener( this.changeCallback );
      }
    },

    /**
     * Attempt to detach listeners from the paint's primary (the paint itself).
     * @private
     *
     * NOTE: If it's a Property, we'll also need to handle the secondary (part inside the Property).
     *
     * @param {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern} paint
     */
    detachPrimary: function( paint ) {
      if ( paint instanceof Property ) {
        paint.unlink( this.updateListener );
        this.detachSecondary( paint.get() );
        this.secondary = null;
      }
      else if ( paint instanceof Color ) {
        paint.removeChangeListener( this.changeCallback );
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
      this.secondary = paint;
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
      this.secondary = null;
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
