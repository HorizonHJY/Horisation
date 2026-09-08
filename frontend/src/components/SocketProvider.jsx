import React, { createContext, useContext, useEffect, useRef, useState } from 'react'
import { io } from 'socket.io-client'

/**
 * One Socket.IO connection for the whole signed-in session.
 *
 * It used to be created inside the Friends page and torn down on unmount, so
 * every realtime event — friend requests, contact requests, incoming messages,
 * trade intents — only arrived while that one page happened to be open.
 * Anywhere else you were blind, and the sidebar badge fell back to a 30s poll.
 *
 * The connection's lifetime now follows the session, not the current view.
 * Pages subscribe to the events they care about and unsubscribe on unmount;
 * they must never disconnect the socket itself.
 */
const SocketContext = createContext({ socket: null, connected: false })

export const useSocket = () => useContext(SocketContext)

export default function SocketProvider({ enabled, children }) {
  const socketRef = useRef(null)
  const [socket, setSocket] = useState(null)
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    if (!enabled) {
      // Signed out: drop the connection so the server stops counting us online.
      if (socketRef.current) {
        socketRef.current.disconnect()
        socketRef.current = null
        setSocket(null)
      }
      setConnected(false)
      return
    }

    const s = io({ withCredentials: true })
    socketRef.current = s
    setSocket(s)

    const onConnect    = () => setConnected(true)
    const onDisconnect = () => setConnected(false)
    s.on('connect', onConnect)
    s.on('disconnect', onDisconnect)

    return () => {
      s.off('connect', onConnect)
      s.off('disconnect', onDisconnect)
      s.disconnect()
      socketRef.current = null
      setSocket(null)
      setConnected(false)
    }
  }, [enabled])

  return (
    <SocketContext.Provider value={{ socket, connected }}>
      {children}
    </SocketContext.Provider>
  )
}

/**
 * Subscribe to one socket event for the lifetime of a component.
 *
 * Keeps the handler in a ref so a re-render does not detach and reattach the
 * listener, which is what makes it safe to pass an inline arrow function.
 */
export function useSocketEvent(event, handler) {
  const { socket } = useSocket()
  const saved = useRef(handler)

  useEffect(() => { saved.current = handler }, [handler])

  useEffect(() => {
    if (!socket || !event) return
    const fn = (...args) => saved.current?.(...args)
    socket.on(event, fn)
    return () => socket.off(event, fn)
  }, [socket, event])
}
