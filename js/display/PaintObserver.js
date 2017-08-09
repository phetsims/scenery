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

    // @private {function} - To be called when a potential change is detected
    this.notifyChangeCallback = this.notifyChanged.bind( this );

    // @private {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern}
    // Our unwrapped fill/stroke value
    this.primary = null;

    // @private {function} - To be called whenever our secondary fill/stroke value may have changed
    this.updateSecondaryListener = this.updateSecondary.bind( this );

    // Tracking needed so we don't add duplicate listeners, see https://github.com/phetsims/axon/issues/129
    // @private {Array.<Property.<*>>} - Indexed the same as the counts.
    this.secondaryListenedProperties = [];
    // @private {Array.<number>}
    this.secondaryListenedCounts = [];
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
     * Calls the change callback, and invalidates the paint itself if it's a gradient.
     * @private
     */
    notifyChanged: function() {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] changed ' + this.node.id + '.' + this.name );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      var primary = this.node[ this.name ];
      if ( primary instanceof Gradient ) {
        primary.invalidateCanvasGradient();
      }
      this.changeCallback();

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
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
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] primary update ' + this.node.id + '.' + this.name );
        sceneryLog && sceneryLog.Paints && sceneryLog.push();

        this.detachPrimary( this.primary );
        this.primary = primary;
        this.attachPrimary( primary );
        this.notifyChangeCallback();

        sceneryLog && sceneryLog.Paints && sceneryLog.pop();
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
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] secondary update ' + this.node.id + '.' + this.name );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      this.detachSecondary( oldPaint );
      this.attachSecondary( newPaint );
      this.notifyChangeCallback();

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
    },

    /**
     * Attempt to attach listeners to the paint's primary (the paint itself), or something else that acts like the primary
     * (properties on a gradient).
     * @private
     *
     * TODO: Note that this is called for gradient colors also
     *
     * NOTE: If it's a Property, we'll also need to handle the secondary (part inside the Property).
     *
     * @param {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern} paint
     */
    attachPrimary: function( paint ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] attachPrimary ' + this.node.id + '.' + this.name );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      if ( paint instanceof Property ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] add Property listener ' + this.node.id + '.' + this.name );
        sceneryLog && sceneryLog.Paints && sceneryLog.push();
        this.secondaryLazyLinkProperty( paint );
        this.attachSecondary( paint.get() );
        sceneryLog && sceneryLog.Paints && sceneryLog.pop();
      }
      else if ( paint instanceof Color ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] add Color listener ' + this.node.id + '.' + this.name );
        paint.addChangeListener( this.notifyChangeCallback );
      }
      else if ( paint instanceof Gradient ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] add Gradient listeners ' + this.node.id + '.' + this.name );
        sceneryLog && sceneryLog.Paints && sceneryLog.push();
        for ( var i = 0; i < paint.stops.length; i++ ) {
          this.attachPrimary( paint.stops[ i ].color );
        }
        sceneryLog && sceneryLog.Paints && sceneryLog.pop();
      }

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
    },

    /**
     * Attempt to detach listeners from the paint's primary (the paint itself).
     * @private
     *
     * TODO: Note that this is called for gradient colors also
     *
     * NOTE: If it's a Property or Gradient, we'll also need to handle the secondaries (part(s) inside the Property(ies)).
     *
     * @param {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern} paint
     */
    detachPrimary: function( paint ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] detachPrimary ' + this.node.id + '.' + this.name );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      if ( paint instanceof Property ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] remove Property listener ' + this.node.id + '.' + this.name );
        sceneryLog && sceneryLog.Paints && sceneryLog.push();
        this.secondaryUnlinkProperty( paint );
        this.detachSecondary( paint.get() );
        sceneryLog && sceneryLog.Paints && sceneryLog.pop();
      }
      else if ( paint instanceof Color ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] remove Color listener ' + this.node.id + '.' + this.name );
        paint.removeChangeListener( this.notifyChangeCallback );
      }
      else if ( paint instanceof Gradient ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] remove Gradient listeners ' + this.node.id + '.' + this.name );
        sceneryLog && sceneryLog.Paints && sceneryLog.push();
        for ( var i = 0; i < paint.stops.length; i++ ) {
          this.detachPrimary( paint.stops[ i ].color );
        }
        sceneryLog && sceneryLog.Paints && sceneryLog.pop();
      }

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
    },

    /**
     * Attempt to attach listeners to the paint's secondary (part within the Property).
     * @private
     *
     * @param {string|Color} paint
     */
    attachSecondary: function( paint ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] attachSecondary ' + this.node.id + '.' + this.name );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      if ( paint instanceof Color ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] add Color listener ' + this.node.id + '.' + this.name );
        paint.addChangeListener( this.notifyChangeCallback );
      }

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
    },

    /**
     * Attempt to detach listeners from the paint's secondary (part within the Property).
     * @private
     *
     * @param {string|Color} paint
     */
    detachSecondary: function( paint ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] detachSecondary ' + this.node.id + '.' + this.name );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      if ( paint instanceof Color ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] remove Color listener ' + this.node.id + '.' + this.name );
        paint.removeChangeListener( this.notifyChangeCallback );
      }

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
    },

    /**
     * Adds our secondary listener to the Property (unless there is already one, in which case we record the counts).
     * @private
     *
     * @param {Property.<*>} property
     */
    secondaryLazyLinkProperty: function( property ) {
      var index = _.indexOf( this.secondaryListenedProperties, property );
      if ( index >= 0 ) {
        this.secondaryListenedCounts[ index ]++;
      }
      else {
        this.secondaryListenedProperties.push( property );
        this.secondaryListenedCounts.push( 1 );
        property.lazyLink( this.updateSecondaryListener );
      }
    },

    /**
     * Removes our secondary listener from the Property (unless there were more than 1 time we needed to listen to it,
     * in which case we reduce the count).
     * @private
     *
     * @param {Property.<*>} property
     */
    secondaryUnlinkProperty: function( property ) {
      var index = _.indexOf( this.secondaryListenedProperties, property );
      this.secondaryListenedCounts[ index ]--;
      if ( this.secondaryListenedCounts[ index ] === 0 ) {
        this.secondaryListenedProperties.splice( index, 1 );
        this.secondaryListenedCounts.splice( index, 1 );
        property.unlink( this.updateSecondaryListener );
      }
    },

    /**
     * Cleans our state (so it can be potentially re-used).
     * @public (scenery-internal)
     */
    clean: function() {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] clean ' + this.node.id + '.' + this.name );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      this.detachPrimary( this.primary );
      this.primary = null;
      this.node = null;

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
    }
  } );

  return PaintObserver;
} );
