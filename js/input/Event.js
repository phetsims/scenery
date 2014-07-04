// Copyright 2002-2014, University of Colorado

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

  scenery.Event = function Event( args ) {
    // ensure that all of the required args are supplied
    assert && assert( args.trail &&
                      args.type &&
                      args.pointer &&
                      args.target, 'Missing required scenery.Event argument' );

    this.handled = false;
    this.aborted = false;

    // {Trail} path to the leaf-most node, ordered list, from root to leaf
    this.trail = args.trail;

    // {String} what event was triggered on the listener
    this.type = args.type;

    // {Pointer}
    this.pointer = args.pointer;

    // raw DOM InputEvent (TouchEvent, PointerEvent, MouseEvent,...)
    this.domEvent = args.domEvent;

    // {Node} whatever node you attached the listener to, or null when firing events on a Pointer
    this.currentTarget = args.currentTarget;

    // {Node} leaf-most node in trail
    this.target = args.target;

    // TODO: add extended information based on an event here?
  };
  var Event = scenery.Event;

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
