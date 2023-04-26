// Copyright 2013-2023, University of Colorado Boulder

/*
 * A pointer is an abstraction that includes a mouse and touch points (and possibly keys). The mouse is a single
 * pointer, and each finger (for touch) is a pointer.
 *
 * Listeners that can be added to the pointer, and events will be fired on these listeners before any listeners are
 * fired on the Node structure. This is typically very useful for tracking dragging behavior (where the pointer may
 * cross areas where the dragged node is not directly below the pointer any more).
 *
 * A valid listener should be an object. If a listener has a property with a Scenery event name (e.g. 'down' or
 * 'touchmove'), then that property will be assumed to be a method and will be called with the Scenery event (like
 * normal input listeners, see Node.addInputListener).
 *
 * Pointers can have one active "attached" listener, which is the main handler for responding to the events. This helps
 * when the main behavior needs to be interrupted, or to determine if the pointer is already in use. Additionally, this
 * can be used to prevent pointers from dragging or interacting with multiple components at the same time.
 *
 * A listener may have an interrupt() method that will attemp to interrupt its behavior. If it is added as an attached
 * listener, then it must have an interrupt() method.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import TProperty from '../../../axon/js/TProperty.js';
import Vector2 from '../../../dot/js/Vector2.js';
import Enumeration from '../../../phet-core/js/Enumeration.js';
import EnumerationValue from '../../../phet-core/js/EnumerationValue.js';
import IOType from '../../../tandem/js/types/IOType.js';
import StringIO from '../../../tandem/js/types/StringIO.js';
import TAttachableInputListener from './TAttachableInputListener.js';
import { EventContext, scenery, SceneryEvent, TInputListener, Trail } from '../imports.js';

export class Intent extends EnumerationValue {
  // listener attached to the pointer will be used for dragging
  public static readonly DRAG = new Intent();

  // listener attached to pointer is for dragging with a keyboard
  public static readonly KEYBOARD_DRAG = new Intent();

  public static readonly enumeration = new Enumeration( Intent, {
    phetioDocumentation: 'entries when signifying Intent of the pointer'
  } );
}

type PointerType = 'pdom' | 'touch' | 'mouse' | 'pen';

export type ActivePointer = {
  point: Vector2;
} & Pointer;

export default abstract class Pointer {

  // The location of the pointer in the global coordinate system.
  public point: Vector2;

  // Each Pointer subtype should implement a "type" field that can be checked against for scenery input.
  public readonly type: PointerType;

  // The trail that the pointer is currently over (if it has yet been registered). If the pointer has not yet registered
  // a trail, it may be null. If the pointer wasn't over any specific trail, then a trail with only the display's
  // rootNode will be set.
  public trail: Trail | null;

  // The subset of Pointer.trail that is Node.inputEnabled. See Trail.getLastInputEnabledIndex() for details. This is
  // kept separately so that it can be detected when inputEnabled changes.
  public inputEnabledTrail: Trail | null;

  // @deprecated Whether this pointer is 'down' (pressed).
  // Will be phased out in https://github.com/phetsims/scenery/issues/803 to something that is specific for the actual
  // mouse/pen button (since this doesn't generalize well to the left/right mouse buttons).
  public isDownProperty: TProperty<boolean>;

  // Whether there is a main listener "attached" to this pointer. This signals that the
  // listener is "doing" something with the pointer, and that it should be interrupted if other actions need to take
  // over the pointer behavior.
  public attachedProperty: TProperty<boolean>;

  // All attached listeners (will be activated in order).
  private readonly _listeners: TInputListener[];

  // Our main "attached" listener, if there is one (otherwise null)
  private _attachedListener: TAttachableInputListener | null;

  // See setCursor() for more information.
  private _cursor: string | null;

  // (scenery-internal) - Recorded and exposed so that it can be provided to events when there
  // is no "immediate" DOM event (e.g. when a node moves UNDER a pointer and triggers a touch-snag).
  public lastEventContext: EventContext | null;

  // A Pointer can be assigned an intent when a listener is attached to initiate or prevent
  // certain behavior for the life of the listener. Other listeners can observe the Intents on the Pointer and
  // react accordingly
  private _intents: Intent[];

  private _pointerCaptured: boolean;

  // Listeners attached to this pointer that clear the this._intent after input in reserveForDrag functions, referenced
  // so they can be removed on disposal
  private _listenerForDragReserve: TInputListener | null;
  private _listenerForKeyboardDragReserve: TInputListener | null;


  // Pointer is not a PhetioObject and not instrumented, but this type is used for
  // toStateObject in Input
  public static readonly PointerIO = new IOType<Pointer>( 'PointerIO', {
    valueType: Pointer,
    toStateObject: ( pointer: Pointer ) => {
      return {
        point: pointer.point.toStateObject(),
        type: pointer.type
      };
    },
    stateSchema: {
      point: Vector2.Vector2IO,
      type: StringIO
    }
  } );

  /**
   * @param initialPoint
   * @param type - the type of the pointer; can different for each subtype
   */
  protected constructor( initialPoint: Vector2, type: PointerType ) {
    assert && assert( initialPoint === null || initialPoint instanceof Vector2 );
    assert && assert( Object.getPrototypeOf( this ) !== Pointer.prototype, 'Pointer is an abstract class' );

    this.point = initialPoint;
    this.type = type;
    this.trail = null;
    this.inputEnabledTrail = null;
    this.isDownProperty = new BooleanProperty( false );
    this.attachedProperty = new BooleanProperty( false );
    this._listeners = [];
    this._attachedListener = null;
    this._cursor = null;
    this.lastEventContext = null;
    this._intents = [];
    this._pointerCaptured = false;
    this._listenerForDragReserve = null;
    this._listenerForKeyboardDragReserve = null;
  }

  /**
   * Sets a cursor that takes precedence over cursor values specified on the pointer's trail.
   *
   * Typically this can be set when a drag starts (and returned to null when the drag ends), so that the cursor won't
   * change while dragging (regardless of what is actually under the pointer). This generally will only apply to the
   * Mouse subtype of Pointer.
   *
   * NOTE: Consider setting this only for attached listeners in the future (or have a cursor field on pointers).
   */
  public setCursor( cursor: string | null ): this {
    this._cursor = cursor;

    return this;
  }

  public set cursor( value: string | null ) { this.setCursor( value ); }

  public get cursor(): string | null { return this.getCursor(); }

  /**
   * Returns the current cursor override (or null if there is one). See setCursor().
   */
  public getCursor(): string | null {
    return this._cursor;
  }

  /**
   * Returns a defensive copy of all listeners attached to this pointer. (scenery-internal)
   */
  public getListeners(): TInputListener[] {
    return this._listeners.slice();
  }

  public get listeners(): TInputListener[] { return this.getListeners(); }

  /**
   * Adds an input listener to this pointer. If the attach flag is true, then it will be set as the "attached"
   * listener.
   */
  public addInputListener( listener: TInputListener, attach?: boolean ): void {
    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( `addInputListener to ${this.toString()} attach:${attach}` );
    sceneryLog && sceneryLog.Pointer && sceneryLog.push();

    assert && assert( listener, 'A listener must be provided' );
    assert && assert( attach === undefined || typeof attach === 'boolean',
      'If provided, the attach parameter should be a boolean value' );

    assert && assert( !_.includes( this._listeners, listener ),
      'Attempted to add an input listener that was already added' );

    this._listeners.push( listener );

    if ( attach ) {
      assert && assert( listener.interrupt, 'Interrupt should exist on attached listeners' );
      this.attach( listener as TAttachableInputListener );
    }

    sceneryLog && sceneryLog.Pointer && sceneryLog.pop();
  }

  /**
   * Removes an input listener from this pointer.
   */
  public removeInputListener( listener: TInputListener ): void {
    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( `removeInputListener to ${this.toString()}` );
    sceneryLog && sceneryLog.Pointer && sceneryLog.push();

    assert && assert( listener, 'A listener must be provided' );

    const index = _.indexOf( this._listeners, listener );
    assert && assert( index !== -1, 'Could not find the input listener to remove' );

    // If this listener is our attached listener, also detach it
    if ( this.isAttached() && listener === this._attachedListener ) {
      this.detach( listener as TAttachableInputListener );
    }

    this._listeners.splice( index, 1 );

    sceneryLog && sceneryLog.Pointer && sceneryLog.pop();
  }

  /**
   * Returns the listener attached to this pointer with attach(), or null if there isn't one.
   */
  public getAttachedListener(): TAttachableInputListener | null {
    return this._attachedListener;
  }

  public get attachedListener(): TAttachableInputListener | null { return this.getAttachedListener(); }

  /**
   * Returns whether this pointer has an attached (primary) listener.
   */
  public isAttached(): boolean {
    return this.attachedProperty.value;
  }

  /**
   * Some pointers are treated differently because they behave like a touch. This is not exclusive to `Touch and touch
   * events though. See https://github.com/phetsims/scenery/issues/1156
   */
  public isTouchLike(): boolean {
    return false;
  }

  /**
   * Sets whether this pointer is down/pressed, or up.
   *
   * NOTE: Naming convention is for legacy code, would usually have pointer.down
   * TODO: improve name, .setDown( value ) with .down =
   */
  public set isDown( value: boolean ) {
    this.isDownProperty.value = value;
  }

  /**
   * Returns whether this pointer is down/pressed, or up.
   *
   * NOTE: Naming convention is for legacy code, would usually have pointer.down
   * TODO: improve name, .isDown() with .down
   */
  public get isDown(): boolean {
    return this.isDownProperty.value;
  }

  /**
   * If there is an attached listener, interrupt it.
   *
   * After this executes, this pointer should not be attached.
   */
  public interruptAttached(): void {
    if ( this.isAttached() ) {
      this._attachedListener!.interrupt(); // Any listener that uses the 'attach' API should have interrupt()
    }
  }

  /**
   * Interrupts all listeners on this pointer.
   */
  public interruptAll(): void {
    const listeners = this._listeners.slice();
    for ( let i = 0; i < listeners.length; i++ ) {
      const listener = listeners[ i ];
      listener.interrupt && listener.interrupt();
    }
  }

  /**
   * Marks the pointer as attached to this listener.
   */
  private attach( listener: TAttachableInputListener ): void {
    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( `Attaching to ${this.toString()}` );

    assert && assert( !this.isAttached(), 'Attempted to attach to an already attached pointer' );

    this.attachedProperty.value = true;
    this._attachedListener = listener;
  }

  /**
   * @returns - Whether the point changed
   */
  public updatePoint( point: Vector2, eventName = 'event' ): boolean {
    const pointChanged = this.hasPointChanged( point );
    point && sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( `pointer ${eventName} at ${point.toString()}` );

    this.point = point;
    return pointChanged;
  }

  /**
   * Sets information in this Pointer for a given pointer down. (scenery-internal)
   *
   @returns - Whether the point changed
   */
  public down( event: Event ): void {
    this.isDown = true;
  }

  /**
   * Sets information in this Pointer for a given pointer up. (scenery-internal)
   *
   * @returns - Whether the point changed
   */
  public up( point: Vector2, event: Event ): boolean {

    this.isDown = false;
    return this.updatePoint( point, 'up' );
  }

  /**
   * Sets information in this Pointer for a given pointer cancel. (scenery-internal)
   *
   * @returns - Whether the point changed
   */
  public cancel( point: Vector2 ): boolean {

    this.isDown = false;

    return this.updatePoint( point, 'cancel' );
  }

  /**
   * Marks the pointer as detached from a previously attached listener.
   */
  private detach( listener: TAttachableInputListener ): void {
    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( `Detaching from ${this.toString()}` );

    assert && assert( this.isAttached(), 'Cannot detach a listener if one is not attached' );
    assert && assert( this._attachedListener === listener, 'Cannot detach a different listener' );

    this.attachedProperty.value = false;
    this._attachedListener = null;
  }

  /**
   * Determines whether the point of the pointer has changed (used in mouse/touch/pen).
   */
  protected hasPointChanged( point: Vector2 ): boolean {
    return this.point !== point && ( !point || !this.point || !this.point.equals( point ) );
  }

  /**
   * Adds an Intent Pointer. By setting Intent, other listeners in the dispatch phase can react accordingly.
   * Note that the Intent can be changed by listeners up the dispatch phase or on the next press. See Intent enum
   * for valid entries.
   */
  public addIntent( intent: Intent ): void {
    assert && assert( Intent.enumeration.includes( intent ), 'trying to set unsupported intent for Pointer' );

    if ( !this._intents.includes( intent ) ) {
      this._intents.push( intent );
    }

    assert && assert( this._intents.length <= Intent.enumeration.values.length, 'to many Intents saved, memory leak likely' );
  }

  /**
   * Remove an Intent from the Pointer. See addIntent for more information.
   */
  public removeIntent( intent: Intent ): void {
    assert && assert( Intent.enumeration.includes( intent ), 'trying to set unsupported intent for Pointer' );

    if ( this._intents.includes( intent ) ) {
      const index = this._intents.indexOf( intent );
      this._intents.splice( index, 1 );
    }
  }

  /**
   * Returns whether or not this Pointer has been assigned the provided Intent.
   */
  public hasIntent( intent: Intent ): boolean {
    return this._intents.includes( intent );
  }

  /**
   * Set the intent of this Pointer to indicate that it will be used for mouse/touch style dragging, indicating to
   * other listeners in the dispatch phase that behavior may need to change. Adds a listener to the pointer (with
   * self removal) that clears the intent when the pointer receives an "up" event. Should generally be called on
   * the Pointer in response to a down event.
   */
  public reserveForDrag(): void {

    // if the Pointer hasn't already been reserved for drag in Input event dispatch, in which
    // case it already has Intent and listener to remove Intent
    if ( !this._intents.includes( Intent.DRAG ) ) {
      this.addIntent( Intent.DRAG );

      const listener = {
        up: ( event: SceneryEvent<TouchEvent | MouseEvent | PointerEvent> ) => {
          this.removeIntent( Intent.DRAG );
          this.removeInputListener( this._listenerForDragReserve! );
          this._listenerForDragReserve = null;
        }
      };

      assert && assert( this._listenerForDragReserve === null, 'still a listener to reserve pointer, memory leak likely' );
      this._listenerForDragReserve = listener;
      this.addInputListener( this._listenerForDragReserve );
    }
  }

  /**
   * Set the intent of this Pointer to indicate that it will be used for keyboard style dragging, indicating to
   * other listeners in the dispatch that behavior may need to change. Adds a listener to the pointer (with self
   * removal) that clears the intent when the pointer receives a "keyup" or "blur" event. Should generally be called
   * on the Pointer in response to a keydown event.
   */
  public reserveForKeyboardDrag(): void {

    if ( !this._intents.includes( Intent.KEYBOARD_DRAG ) ) {
      this.addIntent( Intent.KEYBOARD_DRAG );

      const listener = {
        keyup: ( event: SceneryEvent<KeyboardEvent> ) => clearIntent(),

        // clear on blur as well since focus may be lost before we receive a keyup event
        blur: ( event: SceneryEvent<FocusEvent> ) => clearIntent()
      };

      const clearIntent = () => {
        this.removeIntent( Intent.KEYBOARD_DRAG );
        this.removeInputListener( this._listenerForKeyboardDragReserve! );
        this._listenerForKeyboardDragReserve = null;
      };

      assert && assert( this._listenerForDragReserve === null, 'still a listener on Pointer for reserve, memory leak likely' );
      this._listenerForKeyboardDragReserve = listener;
      this.addInputListener( this._listenerForKeyboardDragReserve );
    }
  }

  /**
   * This is called when a capture starts on this pointer. We request it on pointerstart, and if received, we should
   * generally receive events outside the window.
   */
  public onGotPointerCapture(): void {
    this._pointerCaptured = true;
  }

  /**
   * This is called when a capture ends on this pointer. This happens normally when the user releases the pointer above
   * the sim or outside, but also in cases where we have NOT received an up/end.
   *
   * See https://github.com/phetsims/scenery/issues/1186 for more information. We'll want to interrupt the pointer
   * on this case regardless,
   */
  public onLostPointerCapture(): void {
    if ( this._pointerCaptured ) {
      this.interruptAll();
    }
    this._pointerCaptured = false;
  }

  /**
   * Releases references so it can be garbage collected.
   */
  public dispose(): void {
    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( `Disposing ${this.toString()}` );

    // remove listeners that would clear intent on disposal
    if ( this._listenerForDragReserve && this._listeners.includes( this._listenerForDragReserve ) ) {
      this.removeInputListener( this._listenerForDragReserve );
    }
    if ( this._listenerForKeyboardDragReserve && this._listeners.includes( this._listenerForKeyboardDragReserve ) ) {
      this.removeInputListener( this._listenerForKeyboardDragReserve );
    }

    assert && assert( this._attachedListener === null, 'Attached listeners should be cleared before pointer disposal' );
    assert && assert( this._listeners.length === 0, 'Should not have listeners when a pointer is disposed' );
  }

  public toString(): string {
    return `Pointer#${this.type}_at_${this.point}`;
  }
}

scenery.register( 'Pointer', Pointer );
