// Copyright 2013-2017, University of Colorado Boulder

/**
 * Hooks up listeners to a paint (fill or stroke) to determine when its represented value has changed.
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
   * An observer for a paint (fill or stroke), that will be able to trigger notifications when it changes.
   * @constructor
   *
   * @param {function} changeCallback - To be called on any change (with no arguments)
   */
  function PaintObserver( changeCallback ) {

    // @private {PaintDef}
    // Our unwrapped fill/stroke value
    this.primary = null;

    // @private {function} - Our callback
    this.changeCallback = changeCallback;

    // @private {function} - To be called when a potential change is detected
    this.notifyChangeCallback = this.notifyChanged.bind( this );

    // @private {function} - To be called whenever our secondary fill/stroke value may have changed
    this.updateSecondaryListener = this.updateSecondary.bind( this );

    // @private {Object} - Maps {number} property.id => {number} count (number of times we would be listening to it)
    this.secondaryPropertyCountsMap = {};
  }

  scenery.register( 'PaintObserver', PaintObserver );

  inherit( Object, PaintObserver, {
    /**
     * Should be called when our paint (fill/stroke) may have changed.
     * @public (scenery-internal)
     *
     * Should update any listeners (if necessary), and call the callback (if necessary).
     *
     * NOTE: To clean state, set this to null.
     *
     * @param {PaintDef} primary
     */
    setPrimary: function( primary ) {
      if ( primary !== this.primary ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] primary update' );
        sceneryLog && sceneryLog.Paints && sceneryLog.push();

        this.detachPrimary( this.primary );
        this.primary = primary;
        this.attachPrimary( primary );
        this.notifyChangeCallback();

        sceneryLog && sceneryLog.Paints && sceneryLog.pop();
      }
    },

    /**
     * Releases references without sending the notifications.
     * @public
     */
    clean: function() {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] clean' );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      this.detachPrimary( this.primary );
      this.primary = null;

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
    },

    /**
     * Called when the value of a "primary" Property (contents of one, main or as a Gradient) is potentially changed.
     * @private
     *
     * @param {string|Color} newPaint
     * @param {string|Color} oldPaint
     * @param {Property} property
     */
    updateSecondary: function( newPaint, oldPaint, property ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] secondary update' );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      var count = this.secondaryPropertyCountsMap[ property.id ];
      assert && assert( count > 0, 'We should always be removing at least one reference' );

      for ( var i = 0; i < count; i++ ) {
        this.attachSecondary( newPaint );
      }
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
     * @param {PaintDef} paint
     */
    attachPrimary: function( paint ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] attachPrimary' );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      if ( paint instanceof Property ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] add Property listener' );
        sceneryLog && sceneryLog.Paints && sceneryLog.push();
        this.secondaryLazyLinkProperty( paint );
        this.attachSecondary( paint.get() );
        sceneryLog && sceneryLog.Paints && sceneryLog.pop();
      }
      else if ( paint instanceof Color ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] Color changed to immutable' );

        // We set the color to be immutable, so we don't need to add a listener
        paint.setImmutable();
      }
      else if ( paint instanceof Gradient ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] add Gradient listeners' );
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
     * @param {PaintDef} paint
     */
    detachPrimary: function( paint ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] detachPrimary' );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      if ( paint instanceof Property ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] remove Property listener' );
        sceneryLog && sceneryLog.Paints && sceneryLog.push();
        this.secondaryUnlinkProperty( paint );
        sceneryLog && sceneryLog.Paints && sceneryLog.pop();
      }
      else if ( paint instanceof Gradient ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] remove Gradient listeners' );
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
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] attachSecondary' );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      if ( paint instanceof Color ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] Color set to immutable' );

        // We set the color to be immutable, so we don't need to add a listener
        paint.setImmutable();
      }

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
    },

    /**
     * Calls the change callback, and invalidates the paint itself if it's a gradient.
     * @private
     */
    notifyChanged: function() {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] changed' );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      if ( this.primary instanceof Gradient ) {
        this.primary.invalidateCanvasGradient();
      }
      this.changeCallback();

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
    },

    /**
     * Adds our secondary listener to the Property (unless there is already one, in which case we record the counts).
     * @private
     *
     * @param {Property.<*>} property
     */
    secondaryLazyLinkProperty: function( property ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] secondaryLazyLinkProperty ' + property._id );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      var id = property.id;
      var count = this.secondaryPropertyCountsMap[ id ];
      if ( count ) {
        this.secondaryPropertyCountsMap[ id ]++;
      }
      else {
        this.secondaryPropertyCountsMap[ id ] = 1;
        property.lazyLink( this.updateSecondaryListener );
      }

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
    },

    /**
     * Removes our secondary listener from the Property (unless there were more than 1 time we needed to listen to it,
     * in which case we reduce the count).
     * @private
     *
     * @param {Property.<*>} property
     */
    secondaryUnlinkProperty: function( property ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[PaintObserver] secondaryUnlinkProperty ' + property._id );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      var id = property.id;
      var count = --this.secondaryPropertyCountsMap[ id ];
      assert && assert( count >= 0, 'We should have had a reference before' );

      if ( count === 0 ) {
        delete this.secondaryPropertyCountsMap[ id ];
        if ( !property.isDisposed ) {
          property.unlink( this.updateSecondaryListener );
        }
      }

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
    }
  } );

  return PaintObserver;
} );
