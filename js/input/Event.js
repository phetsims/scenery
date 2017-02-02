// Copyright 2013-2016, University of Colorado Boulder


/*
 * An event in Scenery that has similar event-handling characteristics to DOM events.
 * The original DOM event (if any) is available as event.domEvent.
 *
 * Multiple events can be triggered by a single domEvent, so don't assume it is unique.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  function Event( options ) {
    // ensure that all of the required options are supplied
    assert && assert( options.trail && options.type && options.pointer && options.target,
      'Missing required scenery.Event argument' );

    this.handled = false;
    this.aborted = false;

    // @public {Trail} - Path to the leaf-most node, ordered list, from root to leaf
    this.trail = options.trail;

    // {string} what event was triggered on the listener
    this.type = options.type;

    // {Pointer}
    this.pointer = options.pointer;

    // raw DOM InputEvent (TouchEvent, PointerEvent, MouseEvent,...)
    this.domEvent = options.domEvent;

    // {Node} whatever node you attached the listener to, or null when firing events on a Pointer
    this.currentTarget = options.currentTarget;

    // {Node} leaf-most node in trail
    this.target = options.target;

    // TODO: add extended information based on an event here?
  }

  scenery.register( 'Event', Event );

  inherit( Object, Event, {
    // like DOM Event.stopPropagation(), but named differently to indicate it doesn't fire that behavior on the underlying DOM event
    handle: function() {
      this.handled = true;
    },

    // like DOM Event.stopImmediatePropagation(), but named differently to indicate it doesn't fire that behavior on the underlying DOM event
    abort: function() {
      this.handled = true;
      this.aborted = true;
    }
  } );

  return Event;
} );
