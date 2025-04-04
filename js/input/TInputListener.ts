// Copyright 2021-2025, University of Colorado Boulder

/**
 * The main type interface for input listeners.
 *
 * Refer to Input.ts for documentation on the event types.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import type Bounds2 from '../../../dot/js/Bounds2.js';
import type StrictOmit from '../../../phet-core/js/types/StrictOmit.js';
import type Hotkey from '../input/Hotkey.js';
import type SceneryEvent from '../input/SceneryEvent.js';

export type SceneryListenerFunction<T extends Event = Event> = ( event: SceneryEvent<T> ) => void;
export type SceneryNullableListenerFunction<T extends Event = Event> = ( event: SceneryEvent<T> | null ) => void;

type TInputListener = {
  interrupt?: () => void;
  cursor?: string | null;

  listener?: unknown;

  // Function that returns the Bounds2 for AnimatedPanZoomListener to keep in view during drag input.
  // Bounds are in the global coordinate frame.
  // While dragging, the AnimatedPanZoomListener will try to keep these bounds in view. Intended to be
  // called from a listener attached to a Pointer so that the API is compatible with multi-touch.
  createPanTargetBounds?: ( () => Bounds2 ) | null;

  // Hotkeys that will be available whenever a node with this listener is in a focused trail.
  hotkeys?: Hotkey[];

  ////////////////////////////////////////////////
  //////////////////////////////////////////////
  // Only actual events below here

  focus?: SceneryListenerFunction<FocusEvent>;
  blur?: SceneryListenerFunction<FocusEvent>;
  focusin?: SceneryListenerFunction<FocusEvent>;
  focusout?: SceneryListenerFunction<FocusEvent>;

  keydown?: SceneryListenerFunction<KeyboardEvent>;
  keyup?: SceneryListenerFunction<KeyboardEvent>;

  globalkeydown?: SceneryListenerFunction<KeyboardEvent>;
  globalkeyup?: SceneryListenerFunction<KeyboardEvent>;

  click?: SceneryListenerFunction<MouseEvent>;
  input?: SceneryListenerFunction<Event | InputEvent>;
  change?: SceneryListenerFunction;

  down?: SceneryListenerFunction<MouseEvent | TouchEvent | PointerEvent>;
  mousedown?: SceneryListenerFunction<MouseEvent | PointerEvent>;
  touchdown?: SceneryListenerFunction<TouchEvent | PointerEvent>;
  pendown?: SceneryListenerFunction<PointerEvent>;

  up?: SceneryListenerFunction<MouseEvent | TouchEvent | PointerEvent>;
  mouseup?: SceneryListenerFunction<MouseEvent | PointerEvent>;
  touchup?: SceneryListenerFunction<TouchEvent | PointerEvent>;
  penup?: SceneryListenerFunction<PointerEvent>;

  cancel?: SceneryListenerFunction<TouchEvent | PointerEvent>;
  mousecancel?: SceneryListenerFunction<PointerEvent>;
  touchcancel?: SceneryListenerFunction<TouchEvent | PointerEvent>;
  pencancel?: SceneryListenerFunction<PointerEvent>;

  move?: SceneryListenerFunction<MouseEvent | TouchEvent | PointerEvent>;
  mousemove?: SceneryListenerFunction<MouseEvent | PointerEvent>;
  touchmove?: SceneryListenerFunction<TouchEvent | PointerEvent>;
  penmove?: SceneryListenerFunction<PointerEvent>;

  wheel?: SceneryListenerFunction<WheelEvent>;
  mousewheel?: SceneryListenerFunction<WheelEvent>;
  touchwheel?: SceneryListenerFunction<WheelEvent>;
  penwheel?: SceneryListenerFunction<WheelEvent>;

  enter?: SceneryListenerFunction<MouseEvent | TouchEvent | PointerEvent>;
  mouseenter?: SceneryListenerFunction<MouseEvent | PointerEvent>;
  touchenter?: SceneryListenerFunction<TouchEvent | PointerEvent>;
  penenter?: SceneryListenerFunction<PointerEvent>;

  exit?: SceneryListenerFunction<MouseEvent | TouchEvent | PointerEvent>;
  mouseexit?: SceneryListenerFunction<MouseEvent | PointerEvent>;
  touchexit?: SceneryListenerFunction<TouchEvent | PointerEvent>;
  penexit?: SceneryListenerFunction<PointerEvent>;

  over?: SceneryListenerFunction<MouseEvent | TouchEvent | PointerEvent>;
  mouseover?: SceneryListenerFunction<MouseEvent | PointerEvent>;
  touchover?: SceneryListenerFunction<TouchEvent | PointerEvent>;
  penover?: SceneryListenerFunction<PointerEvent>;

  out?: SceneryListenerFunction<MouseEvent | TouchEvent | PointerEvent>;
  mouseout?: SceneryListenerFunction<MouseEvent | PointerEvent>;
  touchout?: SceneryListenerFunction<TouchEvent | PointerEvent>;
  penout?: SceneryListenerFunction<PointerEvent>;
};

// Exclude all but the actual browser events
export type SupportedEventTypes = keyof StrictOmit<TInputListener, 'interrupt' | 'cursor' | 'listener' | 'createPanTargetBounds'>;

export default TInputListener;